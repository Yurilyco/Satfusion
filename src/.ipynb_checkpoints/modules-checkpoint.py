from typing_extensions import Self
from src.sharp.psit.GPPNN_PSIT import GPPNN as GPPNN_PSIT
from src.sharp.sfdi.GPPNN_SFDI import GPPNN as GPPNN_SFDI
from src.sharp.pnn.DoubleConv2d import DoubleConv2d_PNN
from src.sharp.pnn.pnn import PCNN
from src.sharp.pannet.pannet import PANNet_With_MISR, PANNet_Only_Sharpening
from src.sharp.mamba.panmamba_baseline_finalversion import Net
from src.misr.srcnn import SRCNN
from src.misr.highresnet import RecursiveFusion
from src.misr.trnet import TRNet
from src.misr.rams import RAMS
from src.misr.misr_public_modules import DoubleConv2d
from src.misr.misr_public_modules import PixelShuffleBlock
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F

from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat
import random
from torchvision.transforms import GaussianBlur
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
###############################################################################
# Layers
###############################################################################

class ColorCorrection(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 1x1卷积网络（不改变H/W）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),  # 通道扩展
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1)  # 恢复原始通道
        )

    def forward(self, x):
        """
        输入: x [B,C,H,W] (C通常为3-RGB)
        输出: [B,C,H,W] (H/W不变)
        """
        return x + self.conv(x)  # 残差连接：原始图像 + 学习的色彩偏移

class BicubicUpscaledBaseline(nn.Module):
    """ Bicubic upscaled single-image baseline. """

    def __init__(
        self, input_size, output_size, chip_size, interpolation="bicubic", device=None, **kws
    ):
        """ Initialize the BicubicUpscaledBaseline.

        Parameters
        ----------
        input_size : tuple of int
            The input size.
        output_size : tuple of int
            The output size.
        chip_size : tuple of int
            The chip size.
        interpolation : str, optional
            The interpolation method, by default 'bicubic'.
            Available methods: 'nearest', 'bilinear', 'bicubic'.
        """
        super().__init__()
        assert interpolation in ["bilinear", "bicubic", "nearest"]
        self.resize = Resize(output_size, interpolation=interpolation)
        self.output_size = output_size
        self.input_size = input_size
        self.chip_size = chip_size
        self.lr_bands = np.array(S2_ALL_12BANDS["true_color"]) - 1
        self.mean = JIF_S2_MEAN[self.lr_bands]
        self.std = JIF_S2_STD[self.lr_bands]
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the BicubicUpscaledBaseline.

        Parameters
        ----------
        x : Tensor
            The input tensor (a batch of low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (a single upscaled low-res revisit).
        """
        # If all bands are used, get only the RGB bands for WandB image logging
        if x.shape[2] > 3:
            x = x[:, :, S2_ALL_12BANDS["true_color_zero_index"]]
        # Select the first revisit
        x = x[:, 0, :]

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None, :]

        # Normalisation on the channel axis:
        # Add the mean and multiply by the standard deviation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
        # print(111)

        x += torch.as_tensor(self.mean[None, None, ..., None, None]).to(device)
        x *= torch.as_tensor(self.std[None, None, ..., None, None]).to(device)

        # Convert to float, and scale to [0, 1]:
        x = x.float()
        x /= torch.max(x)
        torch.clamp_(x, 0, 1)

        # Upscale to the output size:
        x = self.resize(x)  # upscale (..., T, C, H_o, W_o)
        return x

class Mask(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block1= nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding="same"),
            nn.Sigmoid(),
        )    
        self.block2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=1,padding="same"),
            nn.Sigmoid()
        )

    def forward(self, ms: Tensor) -> Tensor:
        batch_size, revisits, channels, height, width = ms.shape
        x = ms.view(batch_size * revisits, channels, height, width)
        mask = self.block1(x)  # 确保block1无inplace操作
        sum = mask.sum(dim=[2, 3], keepdim=True)
        res = self.block2(sum)  # 确保block2无inplace操作
        res = res.view(batch_size, revisits, 1, 1, 1)
        avg = ms.mean(dim=1, keepdim=True)
        
        # 显式克隆所有张量
        a = ms.clone()
        b = avg.clone()
        c = res.clone()
        b_expanded = b.repeat(1, revisits, 1, 1, 1)
        c_expanded = c.repeat(1, 1, channels, height, width)
        
        # 安全计算
        result = a * c_expanded + b_expanded * (1 - c_expanded)
        return result
class Mask_Cpx_Conv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block1= nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )    
        self.block2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=1,padding="same"),
            nn.Sigmoid()
        )

    def forward(self, ms: Tensor) -> Tensor:
        batch_size, revisits, channels, height, width = ms.shape
        x = ms.view(batch_size * revisits, channels, height, width)
        mask = self.block1(x)  # 确保block1无inplace操作
        sum = mask.sum(dim=[2, 3], keepdim=True)
        res = self.block2(sum)  # 确保block2无inplace操作
        res = res.view(batch_size, revisits, 1, 1, 1)
        avg = ms.mean(dim=1, keepdim=True)
        
        # 显式克隆所有张量
        a = ms.clone()
        b = avg.clone()
        c = res.clone()
        b_expanded = b.repeat(1, revisits, 1, 1, 1)
        c_expanded = c.repeat(1, 1, channels, height, width)
        
        # 安全计算
        result = a * c_expanded + b_expanded * (1 - c_expanded)
        return result
class MultiTemporalGenerator:
    def __init__(self,
                 n_frames=8,
                 target_size: tuple = (50, 50),
                 kernel_size=5,
                 sigma=0.0,
                 temporal_noise=0.02,
                 temporal_jitter=0.1):
        """
        改进版多时相生成器 (带帧间差异)

        参数:
            temporal_noise: 帧间噪声强度 (默认0.02)
            temporal_jitter: 帧间亮度变化幅度 (默认±10%)
        """
        self.n_frames = n_frames
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.temporal_noise = temporal_noise
        self.temporal_jitter = temporal_jitter
    def safe_random_translate(self, image, max_translation=4):
        """
        仅使用 pad + crop 实现平移，不插值，不引入亮度偏移。
        image: [C, H, W]
        """
        tx = random.randint(-max_translation, max_translation)
        ty = random.randint(-max_translation, max_translation)
        C, H, W = image.shape
        shifted = torch.zeros_like(image)

        if tx >= 0:
            x1_src, x2_src = 0, W - tx
            x1_dst, x2_dst = tx, W
        else:
            x1_src, x2_src = -tx, W
            x1_dst, x2_dst = 0, W + tx

        # 垂直方向
        if ty >= 0:
            y1_src, y2_src = 0, H - ty
            y1_dst, y2_dst = ty, H
        else:
            y1_src, y2_src = -ty, H
            y1_dst, y2_dst = 0, H + ty

        # 源区域 → 目标区域，尺寸严格匹配
        shifted[:, y1_dst:y2_dst, x1_dst:x2_dst] = image[:, y1_src:y2_src, x1_src:x2_src]
        
        return shifted
    def random_translate(self,image, max_translation=4):
        #     # 随机生成平移的步长
            translate_x = random.randint(-max_translation, max_translation)
            translate_y = random.randint(-max_translation, max_translation)

            # 使用 torchvision.transforms.functional.affine 进行平移
            # 不旋转（angle=0），不缩放（scale=1），平移 (translate_x, translate_y)
            image = TF.affine(
                image,
                angle=0,
                translate=(translate_x, translate_y),
                scale=1.0,
                shear=0,
                interpolation=InterpolationMode.BILINEAR,  # 默认为 bilinear
                fill=[0.0, 0.0, 0.0]  # ⚠️ 明确设置填充为黑色
            )
            return image
    def add_random_noise(self,image, noise_factor=0.03):
        """
        向图像添加随机噪声
        """
        std = image.std()
        adjusted_factor = std * noise_factor
        noise = torch.randn_like(image) * adjusted_factor
        noisy_image = image + noise
        # 确保像素值在[0, 1]之间
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        return noisy_image
    def adjust_random_brightness(self,image, brightness_factor_range=(0.9, 1.1)):
        """
        随机调整图像的亮度
        """
        jitter = ColorJitter(brightness=brightness_factor_range)
        return jitter(image)

    def generate(self, y: Tensor) -> Tensor:
        """
        生成带时序变化的低分辨率序列
        输入: [B, 1, C=3, H, W]
        输出: [B, R=8, C=3, h, w] (h=H//scale_factor)
        """
        gaussian_blur = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        # 对 y[:, 0] 通道进行高斯模糊
        x_blurred = gaussian_blur(y[:, 0])
        # 下采样操作
        x_downsampled = F.interpolate(x_blurred, size=(50, 50), mode='bilinear', align_corners=False)
        x_downsampled = x_downsampled.unsqueeze(1)
        # 复制 8 份，变为 [2, 8, 3, 50, 50]
        lr = x_downsampled.repeat(1, 8, 1, 1, 1)
        # 对每张图片应用随机平移
        for i in range(lr.shape[1]):  # x.shape[1] 是 8 (批次中的图片数)
            for j in range(lr.shape[0]):  # x.shape[0] 是 2 (batch size)
                lr[j, i] = self.safe_random_translate(lr[j, i],2)
                lr[j, i] = self.add_random_noise(lr[j, i], noise_factor=self.temporal_noise)
                # 随机调整亮度
                lr[j, i] = self.adjust_random_brightness(lr[j, i], brightness_factor_range=(0.9, 1.1))
        return lr
###############################################################################
# OurModels
###############################################################################
class OurMISR(nn.Module):
    """
    OurMISR model : 实现MISR过程中，除了采样之外的其他步骤
    """

    def __init__(
        self,
        model_MISR,
        msi_channels,
        revisits,
        hidden_channels,
        out_channels,
        kernel_size,
        output_size,
        zoom_factor,
        sr_kernel_size,
        use_artificial_dataset,
        use_sampling_model,
        **kws,
    ) -> None:

        super().__init__()
        self.ourMask = Mask_Cpx_Conv(in_channels=msi_channels)
        self.model_MISR = model_MISR
        self.in_channels = 2 * msi_channels #默认使用参考
        self.revisits = revisits
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.zoom_factor = zoom_factor
        self.sr_kernel_size = sr_kernel_size
        self.use_batchnorm = False
        self.use_artificial_dataset = use_artificial_dataset
        self.use_sampling_model = use_sampling_model


        self.encoder = DoubleConv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            use_batchnorm=self.use_batchnorm,
        )
        
        # Fusion SRCNN
        self.fusion_SRCNN = SRCNN(
            residual_layers=1,
            hidden_channels = self.hidden_channels,
            kernel_size=self.kernel_size,
            revisits = self.revisits,
        )

        # Fusion HighResNet
        self.fusion_HighResNet = RecursiveFusion(
            in_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            revisits=self.revisits,
        )

        # Fusion RAMS
        self.fusion_RAMS = RAMS(
            scale=self.zoom_factor,
            filters=32, # 3D卷积核的个数
            kernel_size=self.kernel_size,
            depth=self.revisits,
            r=8,  # 注意力机制的压缩率
            N=12,  # 残差特征注意力块的数量
            out_channels=self.hidden_channels,
        )

        # Fusion TRNet
        self.fusion_TRNet = TRNet(
            config={
                "encoder": {
                    "in_channels": 6,  # RGB通道输入,这里使用的参考图像，所以是2*3
                    "num_layers": 2,  # 编码器中的残差块数量
                    "kernel_size": 3,  # 卷积核大小
                    "channel_size": 64  # 通道数
                },
                "transformer": {
                    "dim": 64,  # Transformer 的维度
                    "depth": 6,  # Transformer 的层数
                    "heads": 8,  # 多头注意力机制的头数
                    "mlp_dim": 128,  # 前馈网络的隐藏层维度
                    "dropout": 0.1  # Dropout 概率
                },
                "decoder": {
                    "final": {
                        "in_channels": 64,  # 解码器输入通道数
                        "kernel_size": 3  # 卷积核大小
                    }
                }
            },
            out_channels=self.hidden_channels,
        )

        ## Super-resolver (upsampler + renderer)
        self.sr = PixelShuffleBlock(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            sr_kernel_size=self.sr_kernel_size,
            zoom_factor=self.zoom_factor,
            use_batchnorm=self.use_batchnorm,
        )

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )


    def forward(
        self, x: Tensor, pan: Optional[Tensor] = None, y: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_sampling_model:
            x = self.ourMask(x)

        if  self.model_MISR != 'RAMS':
            x = self.compute_and_concat_reference_frame_to_input(x)

        # 先变为[B*R,C,H,W]，之后进行encoder，但是注意RAMS不进行encoder，TRMISR不在这里进行encoder
        batch_size, revisits, channels, height, width = x.shape
        hidden_channels = self.hidden_channels
        if self.model_MISR not in ['RAMS', 'TRNet']:
            x = x.view(batch_size * revisits, channels, height, width)
            x = self.encoder(x)
            # x = x.view(batch_size,revisits,hidden_channels,height,width)
            # x = self.ourMask(x)
            # x = x.view(batch_size * revisits, hidden_channels, height, width)
        '''
        补充采样过程内容
        '''

        # 根据self.model_MISR参数，调整x的结构，为融合做准备
        if self.model_MISR == 'SRCNN':
            x = x.view(batch_size, revisits * hidden_channels, height, width)
        elif self.model_MISR == 'HighResNet':
            x = x.view(batch_size, revisits, hidden_channels, height, width)
        elif self.model_MISR == 'RAMS':
            x = x.view(batch_size, revisits, 3, height, width)
            x = x.permute(0, 2, 1, 3, 4)  # RAMS是3D卷积，需要调整输入，假设输入数据是 (B, T, C, H, W)，调整维度顺序为 (B, C, T, H, W)
        elif self.model_MISR == 'TRNet':
            x = x.view(batch_size, revisits, channels, height, width)

        # 进行融合
        if self.model_MISR == 'SRCNN':
            x = self.fusion_SRCNN(x)
        elif self.model_MISR == 'HighResNet':
            x = self.fusion_HighResNet(x)
        elif self.model_MISR == 'RAMS':
            x = self.fusion_RAMS(x)  # 输入数据是 (B, C, T, H, W)，输出数据是 (B, hidden_channels, H, W)
        elif self.model_MISR == 'TRNet':
            x = self.fusion_TRNet(x, K=50)  # 输出数据是 (B, hidden_channels, H, W)
        '''
        补充其他融合方法
        '''

        # 进行上采样
        x = self.sr(x)

        # 进行H、W的尺度变换
        x = self.resize(x)

        x = x[:, None]

        return x    # [B,1,3,H,W]

    def compute_and_concat_reference_frame_to_input(self, x):
        # Current shape: (batch_size, revisits, channels, height, width)
        reference_frame = self.reference_frame(x).expand_as(x)
        # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
        x = torch.cat([x, reference_frame], dim=-3)
        return x

    def reference_frame(self, x: Tensor) -> Tensor:
        """ Compute the reference frame as the median of all low-res revisits.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).

        Returns
        -------
        Tensor
            The reference frame.
        """
        return x.median(dim=-4, keepdim=True).values


class OurSharpening(nn.Module):
    """
        OurSharpening model : 实现Sharpening过程
    """
    def __init__(
            self,
            model_Sharpening,
            msi_channels,
            pan_channels,
            kernel_size,
            output_size,
            out_channels,
            only_use_oursharpening,
            use_artificial_dataset,
            **kws,
    ) -> None:
        ''' 参数解释
        model_Sharpening: 指明Sharpening使用的模型，例如PNN……
        model_MISR: 指明使用的MISR模型，包括SRCNN、HighResNet……
        in_channels： MISR输出的通道数目
        out_channels: 最终输出的通道数目
        '''
        super().__init__()

        self.only_use_oursharpening=only_use_oursharpening
        self.model_Sharpening = model_Sharpening
        self.msi_channels = msi_channels
        self.pan_channels = pan_channels
        self.kernel_size = kernel_size
        self.use_batchnorm = False
        self.output_size = output_size
        self.out_channels = out_channels
        self.use_artificial_dataset = use_artificial_dataset
        
        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )
        self.resizeY = Resize(
            (50, 50),  # 目标尺寸小于原图 → 缩小
            interpolation="bilinear",
            align_corners=False,
            antialias=True  # 缩小务必开启抗锯齿
        )
        self.pcnn = PCNN(
            in_channels=self.msi_channels+self.pan_channels,  # 多光谱波段数3+1
            n1=64,  # 第一层滤波器数 (默认同论文)
            n2=32,  # 第二层滤波器数 (默认同论文)
            f1=9,  # 第一层卷积核大小
            f2=5,  # 第二层卷积核大小
            f3=5,  # 输出层卷积核大小
        )
        self.pannet_with_MISR = PANNet_With_MISR(
            in_channels=self.msi_channels+self.pan_channels,  # 多光谱波段数3+1
            n1=64,  # 第一层滤波器数 (默认同论文PNN)
            n2=32,  # 第二层滤波器数 (默认同论文PNN)
            f1=9,  # 第一层卷积核大小
            f2=5,  # 第二层卷积核大小
            f3=5,  # 输出层卷积核大小
        )
        # self.pannet_with_MISR = PANNet_With_MISR(
        #     in_channels=self.msi_channels,  # 多光谱波段数3
        #     num_res_blocks=4,
        #     output_size=self.output_size,
        # )
        self.pannet_only_sharpening = PANNet_Only_Sharpening(
            in_channels=self.msi_channels+self.pan_channels,  # 多光谱波段数3+1
            n1=64,  # 第一层滤波器数 (默认同论文PNN)
            n2=32,  # 第二层滤波器数 (默认同论文PNN)
            f1=9,  # 第一层卷积核大小
            f2=5,  # 第二层卷积核大小
            f3=5,  # 输出层卷积核大小
            output_size=self.output_size,
        )
        # self.pannet_only_sharpening = PANNet_Only_Sharpening(
        #     in_channels=self.msi_channels,  # 多光谱波段数3
        #     num_res_blocks=4,
        #     output_size=self.output_size,
        # )
        self.myGPPNN_PSIT = GPPNN_PSIT(
            ms_channels=self.msi_channels,
            pan_channels=self.pan_channels,
            n_feat=16,  # self.hidden_channels
        )
        self.Pan_Mamba = Net(
            base_filter=32  # self.hidden_channels
        )

    def forward(
            self, x: Tensor, pan: Tensor, y: Optional[Tensor] = None,
    ) -> Tensor:
        if self.only_use_oursharpening == False:  # 不是单独使用锐化，则 x 是 MISR 输出的结果
            x = x.squeeze(1)  # 形状变为 [B, C, H, W]
            pan = pan.squeeze(1)  # 形状变为 [B, 1, H, W] C==1

            # 选择锐化模型
            if self.model_Sharpening == 'PNN':
                x = self.pcnn(ms=x, pan=pan)
            if (self.model_Sharpening == 'PANNet'):
                x = self.pannet_with_MISR(ms=x, pan=pan)
            if (self.model_Sharpening == 'Pan_Mamba'):
                # x = self.encoder_pan(x)
                # ms = self.encoder_msi(ms) #undate 3.25
                x = self.Pan_Mamba.forward(ms=x, _=0, pan=pan)
            if (self.model_Sharpening == 'PSIT'):
                x = self.myGPPNN_PSIT(ms=x, pan=pan)
            if (self.model_Sharpening == 'SFDI'):
                x = self.myGPPNN_SFDI(ms=x, pan=pan)

            x = x[:, None]
            return x
        else:  # 单独使用锐化
            pan = pan.squeeze(1)  # 形状变为 [B, 1, H, W]
            i = random.randint(0, x.shape[1]-1)  # 随机选择一个图像
            #print(i)
            x = x[:, i, :, :, :]  # 取出第 1 维的第 i 个元素
            # 经过一个全色锐化的神经网络（输入是 [B, C+1, H, W]，输出是 [B, C, H, W]），进行融合
            if self.model_Sharpening == 'PNN':
                x = self.resize(x)
                x = self.pcnn(ms=x, pan=pan)
            if (self.model_Sharpening == 'PANNet'):
                x = self.pannet_only_sharpening(ms=x, pan=pan)
            if (self.model_Sharpening == 'Pan_Mamba'):
                x = self.resize(x)
                x = self.Pan_Mamba.forward(ms=x, _=0, pan=pan)
            if (self.model_Sharpening == 'PSIT'):
                x = self.resize(x)
                x = self.myGPPNN_PSIT(ms=x, pan=pan)
            x = x[:, None]

            return x


class OurFramework(nn.Module):
    """
        OurFramework model : 实现整个过程，除了采样，因为OurMISR没有实现采样
    """

    def __init__(
            self,
            model_MISR,
            model_Sharpening,
            msi_channels,
            MISR_revisits,
            MISR_hidden_channels,
            MISR_kernel_size,
            output_size,
            out_channels,
            MISR_zoom_factor,
            MISR_sr_kernel_size,
            pan_channels,
            Sharpening_kernel_size,
            use_artificial_dataset,
            use_sampling_model,
            **kws,
    ) -> None:
        super().__init__()
        self.model_MISR = model_MISR
        self.model_Sharpening = model_Sharpening
        self.msi_channels = msi_channels
        self.MISR_revisits = MISR_revisits
        self.MISR_hidden_channels = MISR_hidden_channels
        self.MISR_kernel_size = MISR_kernel_size
        self.output_size = output_size
        self.MISR_zoom_factor = MISR_zoom_factor
        self.MISR_sr_kernel_size = MISR_sr_kernel_size
        self.pan_channels = pan_channels
        self.out_channels = out_channels
        self.Sharpening_kernel_size = Sharpening_kernel_size
        self.use_artificial_dataset = use_artificial_dataset
        self.use_sampling_model = use_sampling_model

        # 创建MISR，实现多时相融合
        self.MISR = OurMISR(
            model_MISR = self.model_MISR,
            msi_channels = self.msi_channels,
            out_channels = out_channels,
            revisits = self.MISR_revisits,
            hidden_channels = self.MISR_hidden_channels,
            kernel_size = self.MISR_kernel_size,
            output_size = self.output_size,
            zoom_factor = self.MISR_zoom_factor,
            sr_kernel_size = self.MISR_sr_kernel_size,
            use_artificial_dataset=self.use_artificial_dataset,
            use_sampling_model=self.use_sampling_model,
        )

        # 创建Sharpening，实现多源融合

        self.Sharpening = OurSharpening(
            model_Sharpening = self.model_Sharpening,
            msi_channels=self.msi_channels,
            pan_channels = self.pan_channels,
            kernel_size=self.Sharpening_kernel_size,
            output_size=self.output_size,
            out_channels = self.out_channels,
            use_artificial_dataset=self.use_artificial_dataset,
            only_use_oursharpening = True if self.model_MISR == "None" else False
        )
        
        self.color_correct = ColorCorrection()
        
    def forward(
            self, x: Tensor, pan: Tensor, y: Optional[Tensor] = None
    ) :
        '''
        if MISR+None :
            return current_out[B, 1, C=3, H, W]、misr_out[B, 1, C=3, H, W]、sharpening_out=None
        elif None+Sharpening :
            return current_out[B, 1, C=3, H, W]、misr_out=None、sharpening_out[B, 1, C=3, H, W]
        elif MISR+Sharpening :
            return current_out[B, 1, C=3, H, W]、misr_out[B, 1, C=3, H, W]、sharpening_out[B, 1, C=3, H, W]
        '''
        # 初始化中间输出，用于分阶段监督，即多阶段损失计算
        misr_out, sharpening_out = None, None

        # Stage 1: 多时相融合（MISR）
        if self.model_MISR != "None":
            misr_out = self.MISR(x=x, y=y)         # current_out = misr_out: [B, 1, C=3, H, W]
            current_out = misr_out
        else:
            current_out = x                 # current_out = x: [B, R, C=3, h, w]

        # Stage 2、3: 全色锐化/多源融合（Sharpening）、组合阶段（当进行 Stage 1 and Stage 2 时，才进行Stage 3 ）
        if self.model_Sharpening != "None" and self.model_MISR != "None":           # MISR+Sharpening
            sharpening_out = self.Sharpening(x=current_out, pan=pan, y=y)
            current_out = current_out + sharpening_out
            # 当MISR+Sharpening时，在最终图像进行颜色调整
            current_out = current_out.squeeze(1)
            current_out = self.color_correct(current_out)
            current_out = current_out[:, None]
        elif self.model_Sharpening != "None" and self.model_MISR == "None":         # None+Sharpening
            sharpening_out = self.Sharpening(x=current_out, pan=pan, y=y)
            # 当None+Sharpening时，在sharpening图像进行颜色调整,当时用人造数据集时，不使用颜色校正代码,使用真实数据集时，进行调整
            if not self.use_artificial_dataset:
                sharpening_out = sharpening_out.squeeze(1)
                sharpening_out = self.color_correct(sharpening_out)
                sharpening_out = sharpening_out[:, None]
            current_out = sharpening_out
        else :                                                                      # MISR+None
            # 当MISR+None时，不进行进行颜色调整
            current_out = current_out
        return current_out, misr_out, sharpening_out


if __name__ == "__main__":

    model_MISR = "RAMS"
    model_Sharpening ="None"
    MISR_in_channels = 3
    MISR_revisits = 8
    MISR_hidden_channels = 128
    MISR_out_channels = 3
    MISR_kernel_size = 3
    output_size = (156,156)
    MISR_zoom_factor = 2
    MISR_sr_kernel_size = 1
    MISR_registration_kind = 'homography'
    MISR_homography_fc_size = 128
    Sharpening_in_channels = 3
    Sharpening_hidden_channels = 128
    out_channels = 3
    Sharpening_kernel_size = 3
    MISR_use_reference_frame = True



    model = OurFramework(
        model_MISR=model_MISR,
        model_Sharpening=model_Sharpening,
        MISR_in_channels=MISR_in_channels,
        MISR_revisits=MISR_revisits,
        MISR_hidden_channels=MISR_hidden_channels,
        MISR_kernel_size=MISR_kernel_size,
        output_size=output_size,
        MISR_zoom_factor=MISR_zoom_factor,
        MISR_sr_kernel_size=MISR_sr_kernel_size,
        MISR_registration_kind=MISR_registration_kind,
        MISR_homography_fc_size=MISR_homography_fc_size,
        Sharpening_in_channels=Sharpening_in_channels,
        Sharpening_hidden_channels=Sharpening_hidden_channels,
        out_channels=out_channels,
        Sharpening_kernel_size=Sharpening_kernel_size,
        MISR_use_reference_frame=MISR_use_reference_frame,
    )

    # 打印模型结构和输出形状
    # print("Model structure:")
    # print(model)

    # 创建输入张量
    batch_size = 2  # 批次大小
    height, width = 50, 50  # 输入图像大小
    high_height, high_width = 156, 156
    x = torch.rand(batch_size, MISR_revisits, MISR_in_channels, height, width, dtype=torch.float32)
    pan = torch.rand(batch_size, 1, 1, high_height, high_width, dtype=torch.float32)
    print(f"x:",x.shape)
    print(f"pan:",pan.shape)

    # 验证输出
    SR = model(x=x,pan=pan)

    print(f"y:",SR.shape)


