# SatFusion: An Efficient Multimodal Fusion Framework for IoT-Oriented Satellite Image Enhancement
Welcome to the official repository of SatFusion,  an efficient multimodal fusion framework for IoT-Oriented satellite image enhancement. This repository provides the implementation of Satfusion.
## Enviroments
cuda 11.8+  
python 3.10+  
torch 2.4.0+  
Set other related enviroments by "pip install -r requirements.txt".  
To run the block of Pan-Mamba , you need to set Vision-Mamba.You can refer to the guidence in [Pan-Mamba](https://github.com/alexhe101/pan-mamba) and this blog [Install Vision Mamba on Linux](https://zhuanlan.zhihu.com/p/687359086).
## Quick Start
Run "Train.py" to train the model;  
Run "Inference.py" to predict.  
Please follow the comment in the code when setting params.  
The process of training is visible on [Weights & Biases](wandb.ai).([more details](https://wandb.ai/quickstart?))
## Reference Code
[INNformer](https://github.com/KevinJ-Huang/PAN_Sharp_INN_Transformer)    
[SFDI](https://github.com/KevinJ-Huang/SFDI)  
[Pan-Mamba](https://github.com/alexhe101/pan-mamba)
## Issues and Contributions
If you encounter any issues or have suggestions for improvement, please feel free to open an issue in the GitHub issue tracker.   
  
We appreciate it for choosing SatFusion for your satellite image enhancement needs!We hope it proves to be a valuable framework.