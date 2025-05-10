from src.train import *
import torch
import gc

default_train_command = [
    # Batch size, gpus, limits
    "python",
    "--batch_size", "16",
    "--gpus", "-1",
    "--max_epochs", "25",
    "--precision", "16",
    "--seed", "431608443",
    "--data_split_seed", "386564310",

    "--model", "ourframework",
    "--ourMISRmodel", "None",
    "--ourSharpeningmodel", "PSIT",
    "--w_mse", "0.3",
    "--w_mae", "0.4",
    "--w_ssim", "0.3",
    "--hidden_channels", "128",
    "--shift_px", "2",
    "--shift_mode", "lanczos",
    "--shift_step", "0.5",
    #"--residual_layers", "1",
    "--learning_rate", "1e-4",

    # Data
    "--dataset", "JIF",
    "--root", "/root/shared-nvme/worldstrat0420/dataset_entire",
    "--revisits", "8",
    "--input_size", "160", "160", 
    "--output_size", "500", "500",
    "--chip_size", "50", "50",
    "--lr_bands_to_use", "true_color",
    # Training, validation, test splits
    "--list_of_aois", "./pretrained_model/split1.csv",
    #--use_artificial_dataset
    "--use_artificial_dataset",
    "--temporal_jitter","0.1",
    "--temporal_noise","0.3"

]
if __name__ == '__main__':
    sys.argv = default_train_command
    sys.argv += ["--num_workers", "16"]

    args = parse_arguments()
    set_random_seed(args)
    initialise_wandb(args)

    dataloaders = load_dataset(args)
    generate_model_backbone(args, dataloaders)
    add_gpu_augmentations(args)
    torch.serialization.add_safe_globals([Compose])
    model = LitModel.load_from_checkpoint(
        checkpoint_path="/root/shared-nvme/worldstrat423/checkpoints/38ryncsh-checkpoint.ckpt",
        backbone = args.backbone,
        weights_only=False
    )
    print(model.hparams) 
    add_callbacks(args, dataloaders)
    trainer = pl.Trainer.from_argparse_args(args)
    # print(trainer)
    trainer.predict(model=model, dataloaders=[ dataloaders["test"]])
    finish_wandb_logging(args)

