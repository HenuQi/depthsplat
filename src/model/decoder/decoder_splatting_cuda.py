from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]

# DecoderSplattingCUDA 模块，使用 CUDA 加速的 splatting 方法进行渲染。
class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )
# DecoderSplattingCUDA 的前向传播流程：
# 输入：
#        gaussians:(B,V,H*W,C_g)，每个像素的
#        extrinsics:(B,V,4,4)，相机外参。
#        intrinsics:(B,V,3,3)，相机内参。
#        near:(B,V)，近裁剪面距离。     
#        far:(B,V)，远裁剪面距离。
#        image_shape:(H,W)，图像的高度和宽度。
# 输出：
#       DecoderOutput(color, depth)
#           color:(B,V,3,H,W)，渲染得到的颜色图。
#           depth:(B,V,H,W)，渲染得到的深度图（如果depth_mode不为None时才输出深度图）。
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput( # DecoderOutput 数据类，用于存储渲染结果。包含 color 和 depth 两个字段，
            color,                  # 渲染图片：(B,V,3,H,W)，每个像素的 RGB 颜色值。
            None
            if depth_mode is None
            else self.render_depth( # （需要的话）渲染深度图 (B,V,H,W),每个像素的深度颜色。
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )


# 渲染深度图的函数，输入与前向传播类似，但输出是深度图。
# 输入：
#        gaussians:(B,V,H*W,C_g)，每个像素的高斯参数。
#        extrinsics:(B,V,4,4)，相机外参         
#        intrinsics:(B,V,3,3)，相机内参。
#        near:(B,V)，近裁剪面距离。     
#        far:(B,V)，远裁剪面距离。
#        image_shape:(H,W)，图像的高度和宽度。
#        mode:DepthRenderingMode，深度渲染模式（如 "depth" 或 "disparity"）。
# 输出：
#       depth:(B,V,H,W)，渲染得到的深度图。
    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
