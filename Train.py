from src.train import *
import torch
import gc
import time
default_train_command = [
    # Batch size, gpus, limits
    "python",
    "--gpus", "-1",
    "--precision", "16",
    # "--seed", "431608443",
    # "--data_split_seed", "386564310",
    "--seed", "122938034",
    "--data_split_seed", "386564310",
    "--batch_size", "4",
    "--num_workers", "32",
    "--max_epochs", "25",

    # Model/Hyperparameters
    "--model", "ourframework",
    "--hidden_channels", "128",
    "--shift_px", "2",
    "--shift_mode", "lanczos",
    "--shift_step", "0.5",
    "--learning_rate", "1e-4",
    "--use_reference_frame",

    # Data
    "--dataset", "JIF",
    "--root", "./dataset_example",
    "--input_size", "160", "160",
    "--output_size", "500", "500",
    "--chip_size", "50", "50",
    "--lr_bands_to_use", "true_color",
    "--revisits", "16",

    # loss
    "--w_mse", "0.3",
    "--w_mae", "0.3",
    "--w_ssim", "0.2",
    "--w_tv", "0.0",
    "--w_sam", "0.2",

    # Training, validation, test splits
    "--list_of_aois", "pretrained_model/final_split.csv",
    
    # do not set below params both None
    "--ourMISRmodel","TRNet", # other choice: None,HighResNet,RAMS,TRNet
    "--ourSharpeningmodel","Pan_Mamba",# other choice: PANNet,PSIT,Pan_Mamba
    # INNformer is PSIT here
]

def run_training_command(training_command):
    sys.argv = training_command
    cli_main()

if __name__ == '__main__':

    ''' 
    ----------------------------------Readme---------------------------------------------------------------------------------------
        # use artificial dataset
        + ["--temporal_noise","0.30"]
        + ["--temporal_jitter","0.15"]
        + ["--use_artificial_dataset"]

        # use filtering
        + ["--use_sampling_model"]
    ----------------------------------Readme---------------------------------------------------------------------------------------
    '''

    run0= [
        default_train_command
    ]

    
    for replicates in [run0]:
        for replicate_training_command in replicates:
            run_training_command(replicate_training_command)
