from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg: # GaussianAdapter的配置类，包含了高斯适配器的相关参数，在config/model/encoder/depthsplat.yaml中配置
    gaussian_scale_min: float       # 1e-10
    gaussian_scale_max: float       # 3.
    sh_degree: int                  # 球谐函数 (Spherical Harmonics, SH) 的阶数：配置为 2，表示使用二阶球谐函数，输出9个球谐系数（包括DC分量和8个高阶分量）。如果配置为 0，则只使用DC分量，输出1个球谐系数。

# GaussianAdapter模块，负责将输入的特征映射到高斯参数空间，包括尺度、旋转、球谐函数等。
# 用于把 raw Gaussian 参数转成世界坐标下的 3D Gaussians
class GaussianAdapter(nn.Module): 
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
    
    # GaussianAdapter高斯适配器模块的前向传播流程：
    # 输入：
    #        extrinsics:(B,V,1,1,1,4,4)，       相机外参，经过重排和扩展后，每个像素都有对应的外参。
    #        intrinsics:(B,V,1,1,1,3,3)，       相机内参，经过重排和扩展后，每个像素都有对应的内参。
    #        xy_ray:(B,V,H*W,1,2)，             每个像素的真实 ray 位置。（每个Gaussian对应的真实 像素位置。）
    #        depths:(B,V,H*W,1,1)，             每个像素的深度值。
    #        opacities:(B,V,H*W,1,1)，          每个像素的不透明度。
    #        gaussians:(B,V,H*W,1,1,C_g-1-2=34)   初始的高斯参数(-1：预测不透明度，-2：预测x,y的偏移量，)。
    #        (h, w)：                           图像的高度和宽度。
    #        input_images:(B,V,3,H,W)           输入图像，用于初始化球谐函数的输入特征。（可选）
    # 输出：
    #       Gaussians(means, covariances, harmonics, opacities, scales, rotations)
    #           高斯位置：means: [B, V, H*W, 1, 1, 3]
    #           协方差：covariances: [B, V, H*W, 1, 1, 3, 3]
    #           轴向尺度：scales: [B, V, H*W, 1, 1, 3]
    #           旋转四元数：rotations: [B, V, H*W, 1, 1, 4]
    #           球谐系数：harmonics: [B, V, H*W, 1, 1, 3, 9]
    #           不透明度：opacities: [B, V, H*W, 1, 1]
    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1) 
        # C_g-1-2=34个参数, 分别是：
        # 3个是尺度参数；4个是旋转参数；d_sh=9，所以3 * d_sh=27个是球谐参数。

    # scales 用 softplus 做非线性保证正值，并 clamp 在最小/最大范围内：
        scales = torch.clamp(F.softplus(scales - 4.), 
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )
        # softplus(x) = log(1+e^x): 确保输出总是正数 (>0)。 以 4 为分界线，输入小于4时输出接近于 0；当输入大于4时，输出接近于 输入值 - 4。
        # clamp 用于限制输出在指定范围内，防止尺度过小或过大导致数值不稳定。
        #       最小值：1e-10
        #       最大值：3

        assert input_images is not None

    # rotations 用 norm 归一化
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        # .norm():计算每个向量的 L2 范数（即：所有元素的平方和 再开根号）    eps:很小的数，防止分母为零。
        # 归一化后的 rotations 是一个单位四元数，表示旋转。

    # 球谐：
        # [2, 2, 65536, 1, 1, 3, 25]  (此时球谐不是2阶)
        # (B, V, H*W, 1, 1, 3, 9)，这是球谐为二阶时。
        # sh reshape 成 [... xyz d_sh] 方便后续操作，并 broadcast 给所有 Gaussian：
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        # sh reshape 成 (B, V, H*W, 1, 1, 3, 9)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        # broadcast_to()：确认 shape 是(B, V, H*W, 1, 1, 3, 9)
        # * self.sh_mask: 乘以 sh_mask 来调整不同阶数球谐系数的初始值，确保 DC 分量较大，高阶分量较小。

        # （默认不走）如果提供了 input_images，用 RGB 初始化第一个 SH 通道，这样 Gaussian 的颜色信息就带入了球谐特征。
        if input_images is not None: 
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

    # 协方差矩阵生成（尺度scales + 旋转rotations）
        # Create world-space covariance matrices.
        # 先计算每个高斯在相机坐标系下的协方差矩阵
        covariances = build_covariance(scales, rotations) 
        # 再从相机外参中取出旋转部分，即 相机到世界的旋转矩阵，用于将局部协方差旋转到世界坐标系下。
        c2w_rotations = extrinsics[..., :3, :3] # c2w_rotations：[B,V,3,3]。
        # 最后计算每个高斯在世界空间中的协方差矩阵。
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        # Σ_world​ = R @ Σ_careme @ R^T

    # Gaussian 均值（mean）计算 --- 通过相机坐标系下的 ray 和深度值计算每个 Gaussian 的位置。
        # Compute Gaussian means.
        # 确定射线方向：屏幕像素 → 世界空间射线
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        # 计算点位置：根据深度和深度值计算，原点+深度值*方向就得到最终的点位置
        means = origins + directions * depths[..., None]

        # 封装输出
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]), # 球谐
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property  # @property用于把一个函数变成“属性”访问，而不是函数调用。
    def d_sh(self) -> int: 
        return (self.cfg.sh_degree + 1) ** 2 
        # d_sh = (sh_degree + 1)^2 = 9
    # SH 的系数数量公式是：#coeff = (degree + 1)^2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh 
        # d_in = 7 + 3 * d_sh = 7 + 3 * 9 = 34
    # d_sh = 每个颜色通道需要的 SH 系数数量，共rgb三个颜色通道，因此要 3 * d_sh 
    # 7 是 3 + 4 ，分别是尺度和旋转参数的数量。


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
