import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
# 将四元数转换为旋转矩阵的函数。
# 输入是一个四元数，输出是对应的旋转矩阵(3x3)。
# rotation_xyzw:(#batch, 4) -> rotation:(#batch, 3, 3)
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

# 相机坐标系下的协方差矩阵生成（尺度scales + 旋转rotations）。
# 即：Σ​ = R S S^T R^T 
# 输入：
#   scales :   代表每个高斯的半轴长度（或标准差）
#   rotations: 代表每个高斯的方向/姿态。
# 输出：
#   covariances: 每个高斯在相机坐标系下的协方差矩阵，描述了高斯在空间中的形状和方向。
def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    # 先将将 scale 向量 [sx, sy, sz] 转成对角矩阵，对角线是 sx, sy, sz，其他位置是0。
    scale = scale.diag_embed() # scale:(#batch, 3) -> (#batch, 3, 3).
    # 再将四元数转换为旋转矩阵
    rotation = quaternion_to_matrix(rotation_xyzw) # rotation_xyzw:(#batch, 4) -> rotation:(#batch, 3, 3)
    # 最后通过 Σ​ = R S S^T R^T 计算高斯在相机坐标系下的协方差矩阵。
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )
