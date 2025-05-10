import itertools
import kornia
import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch import Tensor, nn
from pytorch_msssim import ms_ssim
from src.transforms import lanczos_kernel
eps = torch.finfo(torch.float32).eps
# 计算总变差损失，用于衡量图像的平滑性，TV值越小越好，TV Loss 越小，说明图像越平滑；但不能无限制降低，否则可能导致过度平滑，丢失重要细节。
def tv_loss(x: Tensor) -> Tensor:
    """ Total variation loss.
    The sum of the absolute differences for neighboring pixel-values in the input images.

    Parameters
    ----------
    x : Tensor
        Tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    height, width = x.shape[-2:]
    return (kornia.losses.total_variation(x) / (height * width)).mean(dim=1)

# 计算多尺度结构相似性损失，用于衡量两幅图像的相似程度，MSE 只衡量像素级别的误差，而 MS-SSIM 更关注图像的结构信息，MS-SSIM 越大表示两张图像越相似
def ms_ssim_loss(y_hat, y, window_size):
    """ Multi-Scale Structural Similarity loss.
    See: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int
        The size of the gaussian kernel used in the MS-SSIM calculation to smooth the images.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 1 - ms_ssim(y_hat, y, data_range=1, win_size=window_size, size_average=False)     # 损失函数是希望更小越好，因此用 1 - MS-SSIM 作为 Loss

# 计算结构相似性损失，用于衡量两幅图像在结构上的相似度， SSIM 越大表示图像越相似
def ssim_loss(y_hat, y, window_size=5):
    """ Structural Similarity loss.
    See: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int, optional
        The size of the gaussian kernel used in the SSIM calculation to smooth the images, by default 5.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return kornia.losses.ssim_loss(
        y_hat, y, window_size=window_size, reduction="none"
    ).mean(
        dim=(-1, -2, -3)
    )  # over C, H, W

# 计算均值误差L1损失
def mae_loss(y_hat, y):
    """ Mean Absolute Error (L1) loss.
    Sum of all the absolute differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.l1_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W

# 计算均值平方L2损失
def mse_loss(y_hat, y):
    """ Mean Squared Error (L2) loss.
    Sum of all the squared differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.mse_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W

# 计算峰值信噪比损失
def psnr_loss(y_hat, y):
    """ Peak Signal to Noise Ratio (PSNR) loss.
    The logarithm of base ten of the mean squared error between the label 
    and the output, multiplied by ten.

    In the proper form, there should be a minus sign in front of the equation, 
    but since we want to maximize the PSNR, 
    we minimize the negative PSNR (loss), thus the leading minus sign has been omitted.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 10.0 * torch.log10(mse_loss(y_hat, y))

def fre_loss(y_hat, y):
##    loss_fn = nn.L1Loss().to('cuda')
    y_hatf = torch.fft.fft2(y_hat.float()+1e-8, norm='backward')
    yf = torch.fft.fft2(y.float()+1e-8, norm='backward')
    y_hatf_amp = torch.abs(y_hatf)
    y_hatf_pha = torch.angle(y_hatf)
    yf_amp = torch.abs(yf)
    yf_pha = torch.angle(yf)
    return F.l1_loss(y_hatf_amp, yf_amp, reduction="none").mean(dim=(-1, -2, -3)) + F.l1_loss(y_hatf_pha, yf_pha, reduction="none").mean(dim=(-1, -2, -3))
##    return loss_fn(y_hatf_amp[:,:,:,:], yf_amp)+loss_fn(y_hatf_pha[:,:,:,:], yf_pha)
def ergas(y_hat, y, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""

    B, C, H, W = y.shape
    means_real = y.reshape(B, C, -1).mean(dim=-1)
    mses = ((y_hat - y) ** 2).reshape(B, C, -1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS

    return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())

def sam(y_hat, y, eps=1e-8):
    """SAM for (B, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (y_hat * y).sum(dim=1)
    img1_spectral_norm = torch.sqrt((y_hat**2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((y**2).sum(dim=1))
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=-1 + eps, max=1 - eps)
    loss = 1 - cos_theta
    loss = loss.reshape(loss.shape[0], -1)
    return loss.mean(dim=-1).mean()

def D_lambda1(img_fake: torch.Tensor, img_lm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算光谱失真指数 Dλ（优化版）
    输入：
        img_fake: 融合后的图像 [B, C, H, W]（高分辨率，尺寸可能与 img_lm 不同）
        img_lm: 原始低分辨率多光谱图像 [B, C, h, w]
        eps: 数值稳定性参数
    输出：
        Dλ 值（标量，值越小表示光谱保真度越高）
    """
    # Step 1: 将融合图像插值到低分辨率尺寸（与 img_lm 对齐
    B, C, H_fake, W_fake = img_fake.shape
    _, _, H_lm, W_lm = img_lm.shape
    img_fake_resized = F.interpolate(img_fake, size=(H_lm, W_lm), mode="bicubic", align_corners=False)  # [B, C, h, w]

    # Step 2: 计算像素级光谱角（SAM）
    dot_product = torch.sum(img_fake_resized * img_lm, dim=1)  # [B, h, w]
    norm_fake = torch.norm(img_fake_resized, p=2, dim=1)  # [B, h, w]
    norm_lm = torch.norm(img_lm, p=2, dim=1)  # [B, h, w]

    cosine_sim = dot_product / (norm_fake * norm_lm + eps)
    cosine_sim = torch.clamp(cosine_sim, -1.0 + eps, 1.0 - eps)
    sam_rad = torch.acos(cosine_sim)  # [B, h, w]

    # Step 3: 平均所有像素的 SAM 值作为 Dλ
    return torch.mean(sam_rad)


def D_s1(img_fake: torch.Tensor, img_lm: torch.Tensor, pan: torch.Tensor,
        eps: float = 1e-6) -> torch.Tensor:
    """
    计算空间失真指数 Ds（优化版）
    输入：
        img_fake: 融合后的图像 [B, C, H, W]
        img_lm: 原始低分辨率多光谱图像 [B, C, h, w]
        pan: 全色图像 [B, 1, H_pan, W_pan]
        scale: 全色图像与低分辨率图像的缩放比（例如 PAN 分辨率是 MS 的 4 倍时，scale=4）
        eps: 数值稳定性参数
    输出：
        Ds 值（标量，值越小表示空间细节保留越好）
    """
    # Step 1: 对齐图像尺寸
    pan = torch.squeeze(pan, dim=1)
    # print("pan size:", pan.size())
    B, C = img_fake.shape[0], img_fake.shape[1]

    # 对 img_lm 进行上采样到融合图像尺寸（用于对比）
    img_lm_up = F.interpolate(img_lm, size=(img_fake.shape[2], img_fake.shape[3]),
                              mode="bicubic", align_corners=False)  # [B, C, H, W]

    # 对全色图像插值到融合图像尺寸
    pan_resized = F.interpolate(pan, size=(img_fake.shape[2], img_fake.shape[3]),
                                mode="bicubic", align_corners=False)  # [B, 1, H, W]
    # print("pan_resized size:", pan_resized.size())
    # Step 2: 高通滤波提取空间细节（拉普拉斯算子）
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                    dtype=torch.float32, device=img_fake.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # [C, 1, 3, 3]
    # 定义单通道拉普拉斯核
    laplacian_kernel_pan = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                        dtype=torch.float32, device=img_fake.device)
    laplacian_kernel_pan = laplacian_kernel_pan.view(1, 1, 3, 3)  # 形状为 [1, 1, 3, 3]


    # 对融合图像、上采样的低分辨率图像和全色图像滤波
    detail_fused = F.conv2d(img_fake, laplacian_kernel, padding=1, groups=C)  # [B, C, H, W]
    detail_lm_up = F.conv2d(img_lm_up, laplacian_kernel, padding=1, groups=C)  # [B, C, H, W]
    # 对 pan_resized 进行卷积
    detail_pan = F.conv2d(pan_resized, laplacian_kernel_pan, padding=1)  # 输出形状为 [B, 1, H, W]
    detail_pan = detail_pan.repeat(1, C, 1, 1)  # [B, C, H, W]
    # Step 3: 计算空间细节的相关系数（综合两种对比）
    # 对比1: 融合图像 vs 全色图像的高频细节
    corr_pan = _calc_correlation(detail_fused, detail_pan, eps)

    # 对比2: 融合图像 vs 上采样的低分辨率图像的高频细节（避免过度锐化）
    corr_lm = _calc_correlation(detail_fused, detail_lm_up, eps)

    # 综合两种相关性（加权平均）
    Ds = 1 - 0.5 * (corr_pan + corr_lm)
    # print("D_s", Ds)
    return Ds


def _calc_correlation(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """计算两个张量的通道间平均相关系数"""
    x_flat = x.view(x.shape[0], x.shape[1], -1)  # [B, C, H*W]
    y_flat = y.view(y.shape[0], y.shape[1], -1)

    mean_x = torch.mean(x_flat, dim=2, keepdim=True)  # [B, C, 1]
    mean_y = torch.mean(y_flat, dim=2, keepdim=True)

    cov = torch.mean((x_flat - mean_x) * (y_flat - mean_y), dim=2)  # [B, C]
    std_x = torch.std(x_flat, dim=2, unbiased=False)  # [B, C]
    std_y = torch.std(y_flat, dim=2, unbiased=False)

    corr = cov / (std_x * std_y + eps)  # [B, C]
    return torch.mean(corr)  # 标量



def qnr1(img_fake, img_lm, pan, scale = 3,):
    ds = D_s(img_fake, img_lm, pan)
    dlambda = D_lambda(img_fake, img_lm)
    return (1 - dlambda) * (1 - ds)

def uqi(x, y, block_size=8, eps=1e-12):
    """
    批量计算UQI指数
    x,y: [B, C, H, W]
    返回: [B, C] 每个通道对的UQI值
    """
    B, C, H, W = x.shape

    # 分块处理
    x_blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)  # [B,C,nb,nb,bs,bs]
    y_blocks = y.unfold(2, block_size, block_size).unfold(3, block_size, block_size)

    # 转换为 [B, C, num_blocks, block_size^2]
    x_flat = x_blocks.reshape(B, C, -1, block_size * block_size)
    y_flat = y_blocks.reshape(B, C, -1, block_size * block_size)

    # 计算统计量
    mu_x = x_flat.mean(dim=-1, keepdim=True)  # [B,C,N,1]
    mu_y = y_flat.mean(dim=-1, keepdim=True)

    sigma_x = torch.sqrt(((x_flat - mu_x) ** 2).mean(dim=-1))  # [B,C,N]
    sigma_y = torch.sqrt(((y_flat - mu_y) ** 2).mean(dim=-1))
    sigma_xy = ((x_flat - mu_x) * (y_flat - mu_y)).mean(dim=-1)

    # 计算每个块的UQI并平均
    uqi_per_block = (2 * sigma_xy + eps) / (sigma_x ** 2 + sigma_y ** 2 + eps) * \
                    (2 * mu_x.squeeze() * mu_y.squeeze() + eps) / (mu_x.squeeze() ** 2 + mu_y.squeeze() ** 2 + eps)

    return uqi_per_block.mean(dim=-1)  # [B, C]


def D_lambda(fused,ms,  p=1):
    """
    批处理版本的D_lambda计算（修正版）
    输入:
        ms:    低分辨率多光谱图像 [B, C, H_lr, W_lr]
        fused: 融合后的图像      [B, C, H_hr, W_hr]
        p:     指数参数
    输出:
        D_lambda: 标量值
    """
    B, C, H_hr, W_hr = fused.shape
    device = fused.device

    # Step 1: 上采样低分辨率图像
    ms_upsampled = F.interpolate(ms, size=(H_hr, W_hr), mode='bilinear', align_corners=False)

    # Step 2: 生成所有通道对索引
    indices = list(itertools.combinations(range(C), 2))
    idx1 = torch.tensor([i for i, j in indices], device=device)  # 通道对的第一索引
    idx2 = torch.tensor([j for i, j in indices], device=device)  # 通道对的第二索引
    num_pairs = len(indices)

    # Step 3: 正确合并通道对 [B, num_pairs, 2, H, W]
    fused_pairs = torch.stack([fused[:, idx1], fused[:, idx2]], dim=2)  # [B, num_pairs, 2, H, W]
    ms_pairs = torch.stack([ms_upsampled[:, idx1], ms_upsampled[:, idx2]], dim=2)

    # Step 4: 批量计算UQI
    uqi_fused = uqi(fused_pairs[:, :, 0], fused_pairs[:, :, 1])  # [B, num_pairs]
    uqi_ms = uqi(ms_pairs[:, :, 0], ms_pairs[:, :, 1])

    # Step 5: 构建相似性矩阵
    M1 = torch.ones(B, C, C, device=device)
    M2 = torch.ones(B, C, C, device=device)

    # 使用向量化填充对称位置
    M1[:, idx1, idx2] = uqi_fused
    M1[:, idx2, idx1] = uqi_fused
    M2[:, idx1, idx2] = uqi_ms
    M2[:, idx2, idx1] = uqi_ms

    # Step 6: 计算差异（排除对角线）
    mask = ~torch.eye(C, dtype=torch.bool, device=device)
    diff = (M1 - M2).abs().pow(p)
    # print("d_l",diff[:, mask].mean().pow(1/p))
    return diff[:, mask].mean().pow(1/p)


def D_s(fused, ms, pan, q=1, r=4, ws=7):
    """
    批处理版本的D_S计算
    输入:
        pan:   全色图像 [B, 1, H, W]
        ms:    低分辨率多光谱图像 [B, C, H_lr, W_lr]
        fused: 融合后的图像 [B, C, H, W]
        q:     指数参数
        r:     分辨率比例 (H = H_lr * r)
        ws:    滑动窗口大小
    输出:
        D_S: 标量值
    """
    pan = torch.squeeze(pan, dim=1)
    device = pan.device
    B, C, H_lr, W_lr = ms.shape

    # Step 1: 生成降质全色图像 --------------------------------------------
    # 创建均匀滤波核
    kernel = torch.ones(1, 1, ws, ws, dtype=torch.float64, device=device) / (ws ** 2)

    # 应用卷积（保持维度）
    pan_blur = F.conv2d(pan.double(), kernel, padding=ws // 2, groups=1)

    # 下采样到低分辨率
    pan_degraded = F.interpolate(pan_blur, size=(H_lr, W_lr),
                                 mode='bilinear', align_corners=False)

    # Step 2: 准备UQI计算输入 -------------------------------------------
    # 扩展pan到多通道维度 [B, C, H, W]
    pan_expanded = pan.repeat(1, C, 1, 1)  # 用于HR比较
    pan_degraded_expanded = pan_degraded.repeat(1, C, 1, 1)  # 用于LR比较

    # Step 3: 批量计算UQI ----------------------------------------------
    # 计算融合图像与原始PAN的相似度 [B, C]
    uqi_hr = uqi(fused.double(), pan_expanded)

    # 计算MS与降质PAN的相似度 [B, C]
    uqi_lr = uqi(ms.double(), pan_degraded_expanded)

    # Step 4: 计算差异 -------------------------------------------------
    diff = (uqi_hr - uqi_lr).abs().pow(q)
    # print("d_s",diff.mean(dim=1).pow(1 / q).mean() )
    return diff.mean(dim=1).pow(1 / q).mean()  # 先平均通道再平均批次

def qnr (fused,ms,pan,alpha=1,beta=1,p=1,q=1,r=4,ws=7):
	"""calculates Quality with No Reference (QNR).

	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param alpha: emphasizes relevance of spectral distortions to the overall.
	:param beta: emphasizes relevance of spatial distortions to the overall.
	:param p: parameter to emphasize large spectral differences (default = 1).
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).

	:returns:  float -- QNR.
	"""
	a = (1-D_lambda(fused,ms,p=p))**alpha
	b = (1-D_s(fused,ms,pan,q=q,ws=ws,r=r))**beta
	return a*b

def get_patch(x, x_start, y_start, patch_size):
    """ Get a patch of the input tensor. 
    The patch begins at the coordinates (x_start, y_start) ,
    ends at (x_start + patch_size, y_start + patch_size).

    Parameters
    ----------
    x : Tensor
        Tensor of shape (batch_size, channels, height, width).
    x_start : int
        The x-coordinate of the top left corner of the patch.
    y_start : int
        The y-coordinate of the top left corner of the patch.
    patch_size : int
        The height/width of the (square) patch.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size, channels, patch_size, patch_size).
    """
    return x[..., x_start : (x_start + patch_size), y_start : (y_start + patch_size)]


class Shift(nn.Module):
    """ A non-learnable convolutional layer for shifting. 
    Used instead of ShiftNet.
    """

    def __init__(self, shift_by_px, mode="discrete", step=1.0, use_cache=True):
        """ Initialize the Shift layer.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        mode : str, optional
            The mode of shifting, by default 'discrete'.
        step : float, optional
            The step size of the shift, by default 1.0.
        use_cache : bool, optional
            Whether to cache the shifts, by default True.
        """
        super().__init__()
        self.shift_by_px = shift_by_px
        self.mode = mode
        if mode == "discrete":
            shift_kernels = self._shift_kernels(shift_by_px)
        elif mode == "lanczos":
            shift_kernels = self._lanczos_kernels(shift_by_px, step)
        self._make_shift_conv2d(shift_kernels)
        self.register_buffer("shift_kernels", shift_kernels)
        self.y = None
        self.y_hat = None
        self.use_cache = use_cache
        self.shift_cache = {}

    def _make_shift_conv2d(self, kernels):
        """ Make the shift convolutional layer.

        Parameters
        ----------
        kernels : torch.Tensor
            The shift kernels.
        """
        self.number_of_kernels, _, self.kernel_height, self.kernel_width = kernels.shape
        self.conv2d_shift = nn.Conv2d(
            in_channels=self.number_of_kernels,
            out_channels=self.number_of_kernels,
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=False,
            groups=self.number_of_kernels,
            padding_mode="reflect",
        )

        # Fix (kN, 1, kH, kW)
        self.conv2d_shift.weight.data = kernels
        self.conv2d_shift.requires_grad_(False)  # Freeze

    @staticmethod
    def _shift_kernels(shift_by_px):
        """ Create the shift kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.

        Returns
        -------
        torch.Tensor
            The shift kernels.
        """
        kernel_height = kernel_width = (2 * shift_by_px) + 1
        kernels = torch.zeros(
            kernel_height * kernel_width, 1, kernel_height, kernel_width
        )
        all_xy_positions = list(
            itertools.product(range(kernel_height), range(kernel_width))
        )

        for kernel, (x, y) in enumerate(all_xy_positions):
            kernels[kernel, 0, x, y] = 1
        return kernels

    @staticmethod
    def _lanczos_kernels(shift_by_px, shift_step):
        """ Create the Lanczos kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        shift_step : float
            The step size of the shift.

        Returns
        -------
        torch.Tensor
            The Lanczos kernels.
        """
        shift_step = float(shift_step)
        shifts = torch.arange(-shift_by_px, shift_by_px + shift_step, shift_step)
        shifts = shifts[:, None]
        kernel = lanczos_kernel(shifts, kernel_lobes=3)
        kernels = torch.stack(
            [
                kernel_y[:, None] @ kernel_x[None, :]
                for kernel_y, kernel_x in itertools.product(kernel, kernel)
            ]
        )
        return kernels[:, None]

    def forward(self, y: Tensor) -> Tensor:
        """ Forward shift pass.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The shifted tensor.
        """
        batch_size, input_channels, input_height, input_width = y.shape
        patch_height = patch_width = y.shape[-1] - self.kernel_width + 1

        number_of_kernels_dimension = -3

        # TODO: explain what is going on here
        y = y.unsqueeze(dim=number_of_kernels_dimension).expand(
            -1, -1, self.number_of_kernels, -1, -1
        )  # (B, C, kN, H, W)

        y = y.view(
            batch_size * input_channels,
            self.number_of_kernels,
            input_height,
            input_width,
        )

        # Current input shape: (number_of_kernels, batch_channels, height, width)
        y = self.conv2d_shift(y)
        batch_size_channels, number_of_kernels, height, width = 0, 1, 2, 3

        # Transposed input shape: (number_of_kernels, batch_size_channels, height, width)
        y = y.transpose(number_of_kernels, batch_size_channels)

        y = y.contiguous().view(
            self.number_of_kernels * batch_size,
            input_channels,
            patch_height,
            patch_width,
        )
        return y

    def prep_y_hat(self, y_hat):
        """ Prepare the y_hat for the shift by

        Parameters
        ----------
        y_hat : torch.Tensor
            The output tensor.

        Returns
        -------
        Tensor
            ???
        """
        patch_width = y_hat.shape[-1] - self.kernel_width + 1

        # Get center patch
        y_hat = get_patch(
            y_hat,
            x_start=self.shift_by_px,
            y_start=self.shift_by_px,
            patch_size=patch_width,
        )

        # (number_of_kernels, batch_size, channels, height, width)
        y_hat = y_hat.expand(self.number_of_kernels, -1, -1, -1, -1)
        _, batch_size, channels, height, width = y_hat.shape

        # (number_of_kernels*batch_size, channels, height, width)
        return y_hat.contiguous().view(
            self.number_of_kernels * batch_size, channels, height, width
        )

    @staticmethod
    def gather_shifted_y(y: Tensor, ix) -> Tensor:
        """ Gather the shifted y.

        Parameters
        ----------
        y : Tensor
            The input tensor.
        ix : Tensor
            ???

        Returns
        -------
        Tensor
            The shifted y.
        """
        batch_size = ix.shape[0]
        # TODO: Check if 1st dimension is number of kernels
        number_of_kernels_batch_size, channels, height, width = y.shape
        number_of_kernels = number_of_kernels_batch_size // batch_size
        ix = ix[None, :, None, None, None].expand(-1, -1, channels, height, width)

        # (batch_size, channels, height, width)
        return y.view(number_of_kernels, batch_size, channels, height, width).gather(
            dim=0, index=ix
        )[0]

    @staticmethod
    def _hash_y(y):
        """ Hashes y by [???].

        Parameters
        ----------
        y : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The hashed tensor.
        """
        batch_size = y.shape[0]
        return [tuple(row.tolist()) for row in y[:, :4, :4, :4].reshape(batch_size, -1)]

    def registered_loss(self, loss_function):
        """ Creates a loss function adjusted for registration errors 
        by computing the min loss across shifts of up to `self.shift_by_px` pixels.

        Parameters
        ----------
        loss_function : Callable
            The loss function.

        Returns
        -------
        Callable
            The loss function adjusted for registration errors.

        """

        def _loss(y_hat, y=None, **kws):
            """ Compute the loss.

            Parameters
            ----------
            y_hat : Tensor
                The output tensor.
            y : Tensor, optional
                The target tensor, by default None.

            Returns
            -------
            Tensor
                The loss.
            """

            hashed_y = self._hash_y(y)
            cached_y = (
                torch.Tensor([hash in self.shift_cache for hash in hashed_y])
                .bool()
                .to(y.device)
            )
            not_cached_y = ~cached_y

            # If y and y_hat are both cached, return the loss
            if self.y is not None and self.y_hat is not None:
                min_loss = loss_function(self.y_hat, self.y, **kws)
            else:
                batch_size, channels, height, width = y.shape
                min_loss = torch.zeros(batch_size).to(y.device)
                patch_width = width - self.kernel_width + 1  # patch width

                y_all = torch.zeros(batch_size, channels, patch_width, patch_width).to(
                    y.device
                )

                y_hat_all = torch.zeros(
                    batch_size, channels, patch_width, patch_width
                ).to(y.device)

                # If there are any hashes in cache
                if any(cached_y):

                    ix = torch.stack(
                        [
                            self.shift_cache[hash]
                            for hash in hashed_y
                            if hash in self.shift_cache
                        ]
                    )

                    optimal_shift_kernel = self.shift_kernels[ix]
                    print(optimal_shift_kernel.shape)
                    (
                        batch_size,
                        number_of_kernels,
                        _,
                        kernel_height,
                        kernel_width,
                    ) = optimal_shift_kernel.shape

                    conv2d_shift = nn.Conv2d(
                        in_channels=number_of_kernels,
                        out_channels=number_of_kernels,
                        kernel_size=(kernel_height, kernel_width),
                        bias=False,
                        groups=number_of_kernels,
                        padding_mode="reflect",
                    )

                    # Fix and freeze (kN, 1, kH, kW)
                    conv2d_shift.weight.data = optimal_shift_kernel
                    conv2d_shift.requires_grad_(False)
                    y_in = conv2d_shift(y[cached_y].transpose(-3, -4)).transpose(-4, -3)

                    y_hat_in = get_patch(
                        y_hat[cached_y],
                        x_start=self.shift_by_px,
                        y_start=self.shift_by_px,
                        patch_size=patch_width,
                    )  # center patch

                    min_loss[cached_y] = loss_function(y_hat_in, y_in, **kws)
                    y_all[cached_y] = y_in.to(y_all.dtype)
                    y_hat_all[cached_y] = y_hat_in.to(y_hat_all.dtype)

                # If there are any hashes not in cache
                if any(not_cached_y):
                    y_out, y_hat_out = y[not_cached_y], y_hat[not_cached_y]
                    batch_size = y_out.shape[0]
                    y_out = self(y_out)  # (Nbatch, channels, height, width)
                    # (Nbatch, channels, height, width)
                    y_hat_out = self.prep_y_hat(y_hat_out)
                    losses = loss_function(y_hat_out, y_out, **kws).view(
                        -1, batch_size
                    )  # (N, B)
                    min_loss[not_cached_y], ix = torch.min(
                        losses, dim=0
                    )  # min over patches (B,)
                    y_out = self.gather_shifted_y(
                        y_out, ix
                    )  # shifted y (batch_size, channels, height, width)
                    batch_size, channels, height, width = y_out.shape
                    # (batch_size, channels, height, width). Copied along dim 0
                    y_hat_out = y_hat_out.view(-1, batch_size, channels, height, width)

                    y_hat_out = y_hat_out[0]

                    y_all[not_cached_y] = y_out.to(y_all.dtype)
                    y_hat_all[not_cached_y] = y_hat_out.to(y_hat_all.dtype)
                    if self.use_cache:
                        hashed_y = [
                            hash for hash in hashed_y if hash not in self.shift_cache
                        ]
                        for hash, index in zip(hashed_y, ix):
                            self.shift_cache[hash] = ix

                self.y, self.y_hat = y_all, y_hat_all

            return min_loss

        return _loss
