# SatFusion: An Efficient Multimodal Fusion Framework for IoT-Oriented Satellite Image Enhancement
Welcome to the official repository of SatFusion,an efficient multimodal fusion framework for IoT-Oriented satellite image enhancement. This repository provides the implementation of Satfusion.
## Introduction
![](https://cdn.luogu.com.cn/upload/image_hosting/zltsbcxa.png)
## Quick Start
### Code Structure
```
SatFusion/
├── Inference.py
├── Train.py
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── datasources.py
    ├── lightning_modules.py
    ├── loss.py
    ├── modules.py
    ├── plot.py
    ├── train.py
    ├── transforms.py
    ├── misr/
    │   ├── highresnet.py
    │   ├── misr_public_modules.py
    │   ├── rams.py
    │   ├── srcnn.py
    │   └── trnet.py
    └── sharp/
        ├── mamba/
        ├── pannet/
        ├── psit/
        └── sfdi/
```
The files in forders *src/misr* and *src/sharp* are sub-modules of MISR and Pan-Sharpenning respectively.The file *src/modules.py* is our backbone code.Follow the guidance below for better use of SatFusion.
### Enviroments
CUDA 11.8+  
Python 3.10+  
PyTorch 2.4.0+  
Install additional dependencies by running: 
```
pip install -r requirements.txt  
```
To run the block of Pan-Mamba , Vision-Mamba is required.You can refer to the guidance in [Pan-Mamba](https://github.com/alexhe101/pan-mamba) and this blog [Install Vision Mamba on Linux](https://zhuanlan.zhihu.com/p/687359086).
### Dataset
The dataset we used is Worldstrat.Fetch the entire dataset on [https://worldstrat.github.io/](https://worldstrat.github.io/).Here,we provide an example dataset to test the project as *dataset_example*.The files in *pretrained_model* lists the pictures involving our framework for different conditions.
### Train
Set the params $root$ as your root dir of the dataset and $list\_of\_aios$ as "pretrained_model/final_split.csv" in file *Train.py*.Run *Train.py* to train the model;    
The process of training is visible on [Weights & Biases](wandb.ai).For details, refer to [Weights & Biases quickstart guide](https://wandb.ai/quickstart?).
### Inference
Set $list\_of\_aios$ as "pretrained_model/predict_split.csv" or replace it with the aios you want.Load the results of training from folder *checkpoints* and set $checkpoint\_path$ in *Inference.py* as its path.Ensure all other parameters remain consistent with the training configuration.
## Issues and Contributions
If you encounter any issues or have suggestions for improvement, please feel free to open an issue in the GitHub issue tracker.   
  
We appreciate your use of SatFusion for your satellite image enhancement needs!We hope it proves to be a valuable framework.