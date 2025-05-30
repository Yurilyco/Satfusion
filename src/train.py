#!/usr/bin/env python
import sys
import os
import torch
# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Get the root repository directory by going up one level from the script directory
repo_root = os.path.dirname(script_dir)

# Add the repository root to the Python path
sys.path.append(repo_root)
import pandas as pd
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import Compose
from src.datasets import load_dataset_SN7, load_dataset_JIF, load_dataset_ProbaV
from src.lightning_modules import LitModel, ImagePredictionLogger
from src.modules import OurFramework as ourframework
from src.transforms import FilterData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# from pl_bolts.callbacks import ModuleDataMonitor


def cli_main():
    """ CLI entrypoint for training.
    - Parses the CLI arguments
    - Intialises WandB
    - Loads the dataset
    - Generates the model
    - Runs the training
    - Finishes WandB logging
    """

    args = parse_arguments()
    set_random_seed(args)
    initialise_wandb(args)

    dataloaders = load_dataset(args)
    generate_model_backbone(args, dataloaders)
    params = count_parameters(vars(args)["backbone"])
    print(f"{args.ourMISRmodel+ ' + ' + args.ourSharpeningmodel:<20s}| {params:.2f}K") 
    
    add_gpu_augmentations(args)

    model = generate_model(args)

    add_callbacks(args, dataloaders)
    generate_and_run_trainer(args, dataloaders, model)
    finish_wandb_logging(args)

def count_parameters(model):
    """Count the number of trainable parameters (unit: millions)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3  
    
def generate_model(args):
    """ Generates a Lightning model from the arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.

    Returns
    -------
    LightningModule
        The Lightning model.
    """
    model = LitModel(**vars(args))  # Lightning model
    return model


def set_random_seed(args):
    """ Sets the random seed for reproducibility.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing the random seed (args.seed).
    """
    pl.seed_everything(args.seed)


def finish_wandb_logging(args):
    """ Finishes logging in WandB.
    Uploads the model to WandB if the upload_checkpoint flag is set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing the random seed (args.seed).
    """
    if args.upload_checkpoint:
        wandb.save(f"checkpoints/{wandb.run.id}-checkpoint.ckpt")
    wandb.finish()


def generate_and_run_trainer(args, dataloaders, model):
    """ Generates and runs the Lightning trainer.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    dataloaders : dict
        Dictionary containing the dataloaders for the training, validation and
        test set.
    model : LightningModule
        The Lightning model.
    """
    trainer = pl.Trainer.from_argparse_args(args)
    #print(trainer)
    trainer.fit(model, dataloaders["train"], dataloaders["val"])

    if not args.fast_dev_run:
        trainer.test(model=model, dataloaders=dataloaders["test"])


def add_callbacks(args, dataloaders):
    """ Adds logging and early stopping callbacks to the Lightning trainer.
    WandbLogger, ImagePredictionLogger and LearningRateMonitor are added.

    ModelCheckpoint is used to save the best model during training.
    EarlyStopping is used to stop training if the validation loss does not
    improve for 4 epochs.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    dataloaders : dict
        Dictionary containing the dataloaders for the training, validation and
        test set.
    """
    vars(args)["logger"] = WandbLogger(
        project="your_project_name", entity="your_entity_name", config=args
    )


    vars(args)["callbacks"] = [
        ImagePredictionLogger(
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            test_dataloader=dataloaders["test"],
            log_every_n_epochs=1,
            window_size=tuple(args.chip_size),
            revisits=args.revisits,
            use_artificial_dataset=args.use_artificial_dataset,
            temporal_noise=args.temporal_noise,
            temporal_jitter=args.temporal_jitter,
        ),
        LearningRateMonitor(logging_interval="step"),
        #EarlyStopping(monitor="val/loss", patience=4),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{wandb.run.id}-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val/loss",
            mode="min",
        ),
    ]


def add_gpu_augmentations(args):
    """ Adds GPU augmentations to the Lightning model.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    """
    vars(args)["transform"] = Compose(
        [  # GPU augmentations
            #FilterData(values=(0,), thres=0.4, fill="zero", batched=True),
            #RandomRotateFlipDict(angles=[0, 90, 180, 270], batched=True),
        ]
    )


def generate_model_backbone(args, dataloaders):
    """ Generates the model backbone from the arguments and the dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    dataloaders : dict
        Dictionary containing the dataloaders for the training, validation and
    """

    # 改变建立的backbone主干网络：
    msi_channels = dataloaders["train"].dataset[0]["lr"].shape[1]
    out_channels = dataloaders["train"].dataset[0]["hr"].shape[1]
    pan_channels = 1
    crop_ratio = args.chip_size[0] / args.input_size[0]
    output_height, output_width = args.output_size
    cropped_output_size = (
        round(output_height * crop_ratio),
        round(output_width * crop_ratio),
    )

    model_cls = eval(args.model)
    vars(args)["backbone"] = model_cls(
        model_MISR=args.ourMISRmodel,
        model_Sharpening=args.ourSharpeningmodel,
        msi_channels= msi_channels,
        MISR_revisits=args.revisits,
        MISR_hidden_channels=args.hidden_channels,
        MISR_kernel_size=args.kernel_size,
        output_size=cropped_output_size,
        MISR_zoom_factor=args.zoom_factor,
        MISR_sr_kernel_size=args.sr_kernel_size,
        pan_channels=pan_channels,
        Sharpening_kernel_size=args.kernel_size,
        out_channels = out_channels,
        use_artificial_dataset=args.use_artificial_dataset,
        use_sampling_model=args.use_sampling_model,
    )



def load_dataset(args):
    """ Loads the appropriate dataset into dataloaders 
    based on the passed dataset argument.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.

    Returns
    -------
    dict
        Dictionary containing the dataloaders for the training, validation and
        test set.

    Raises
    ------
    Exception
        If the passed dataset name doesn't exist.
    Exception
        If the dataset root is missing.
    """
    if args.dataset == "JIF":
        dataloaders = load_dataset_JIF(**vars(args))
    elif args.dataset == "SN7":
        dataloaders = load_dataset_SN7(**vars(args))
    elif args.dataset == "PROBAV":
        dataloaders = load_dataset_ProbaV(**vars(args))
    else:
        raise Exception("Undefined dataset. Can't create dataloaders.")

    if args.root is None:
        raise Exception("The root path of the dataset needs to be set.")
    return dataloaders


def initialise_wandb(args):
    """ Initialises WandB for logging.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from the CLI and model and project specific arguments.
    """

    tags = [] if not args.upload_checkpoint else ["inference"]
    wandb.init(project="your_project_name", tags=tags,entity="your_entity_name")

    wandb.run.log_code("./src/")


def parse_arguments():
    """ Parses the arguments passed from the CLI using ArgumentParser and
    adds the model and project specific arguments.

    If a list of aois is passed, it is loaded and added to the args namespace.

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed arguments.
    """
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)
    parser = add_project_specific_arguments(parser) #<class 'argparse.ArgumentParser'>
    args = parser.parse_args()
    args = add_list_of_aois_to_args(args)
    return args #<class 'argparse.Namespace'>


def add_list_of_aois_to_args(args):
    """ Adds the list of aois to the args namespace.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the parsed arguments.

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed arguments.
    """
    if args.list_of_aois is not None:
        args.list_of_aois = list(pd.read_csv(args.list_of_aois, index_col=1).index)
    return args


def add_project_specific_arguments(parser):
    """ Adds the project specific arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add the project specific arguments to.

    Returns
    -------
    argparse.ArgumentParser
        Argument parser with the project specific arguments added.
    """

    # Dataset arguments
    parser.add_argument(
        "--scene_classification_filter_values", nargs="+",default=None, type=int
    )
    parser.add_argument(
        "--dataset", default="JIF", type=str, help="JIF or SN7 dataset."
    )
    parser.add_argument(
        "--root", default=None, type=str, help="Root folder of the dataset."
    )
    parser.add_argument(
        "--train_split",
        default=None,
        type=int,
        help="Number of scenes in the train set.",
    )
    parser.add_argument(
        "--val_split", default=None, type=int, help="Number of scenes in the val set."
    )
    parser.add_argument(
        "--test_split", default=None, type=int, help="Number of scenes in the test set."
    )
    parser.add_argument(
        "--list_of_aois",
        default=None,
        type=str,
        help="Pandas DataFrame containing a list of AOIs from the dataset to use.",
    )
    parser.add_argument(
        "--lr_bands_to_use",
        default="all",
        type=str,
        help="Low resolution bands to use: all or true_color. Default: all.",
    )
    parser.add_argument(
        "--radiometry_depth",
        default=12,
        type=int,
        help="Radiometry depth to use: 12 or 8 bits. Default: 12 (full).",
    )
    parser.add_argument(
        "--data_split_seed",
        default=42,
        type=int,
        help="Separate seed to ensure the train/val/test split remains the same.",
    )
    parser.add_argument(
        "--pansharpen_hr",
        default=False,
        type=bool,
        help="Pansharpen the highres RGB using the panchromatic channel.",
    )
    parser.add_argument(
        "--compute_median_std",
        default=False,
        type=bool,
        help="Calculates the mean and std of the input dataset.",
    )

    # Training arguments
    parser.add_argument(
        "--seed", default=1337, type=int, help="Randomization seed."
    )  # random seed
    parser.add_argument("--num_workers", default=8, type=int)  # CPU cores
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--subset_train",
        default=1.0,
        type=float,
        help="Fraction of the training dataset.",
    )

    # Logging arguments
    parser.add_argument(
        "--upload_checkpoint",
        default=False,
        type=bool,
        help="Uploads the model checkpoint to WandB.",
    )
    return parser


# TODO
# - UNet-type encoding (multi-scale bottlenecks)
# - cleanest as reference frame
# - filter revisits based on timestamps
# - log shifts histogram
# - notebook for finding the most similar LRs to an HR
# - whole-image model
# - loss: multi-class CE loss
# - differentiable phase correlation for registered loss? grid search doesn't scale well
# - tiny regularization on TV loss
# - Lanczos resizing

if __name__ == "__main__":
    r"""
    # Usage
    src/train.py 
    # Model arguments
    --batch_size 48 --gpus -1 --precision 16 --model highresnet     /
    --residual_layers 1 --hidden_channels 128                       /
    --shift_px 2 --shift_mode lanczos --shift_step 0.5              /

    # Training arguments
    --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --learning_rate 1e-4       /

    # Dataset arguments
    --dataset JIF --root ./data/ --revisits 8                       /
    --input_size 160 160 --output_size 500 500 --chip_size 50 50    /
    --radiometry_depth 8 --lr_bands all                             /
    --train_split 2700 --val_split 350 --test_split 1               /

    # Reproducibility arguments
    --list_of_aois full.csv                                         /
    --data_split_seed 43 --seed 43                                  /
    --upload_checkpoint True                                        /


    --gpus -1 / 0,1 / 0 or skip     # GPUs all / specific / none
    --fast_dev_run                  # Debugging
    --stochastic_weight_avg         # Stochastic Weight Averaging
    --help                          # Help
    """
    cli_main()
