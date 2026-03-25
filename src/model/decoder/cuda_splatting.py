from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points

# 生成透视投影矩阵 [B, 4, 4]
def get_projection_matrix(
    near: Float[Tensor, " batch"],  # near：近裁剪面
    far: Float[Tensor, " batch"],   # far：远裁剪面
    fov_x: Float[Tensor, " batch"], # fov_x：水平视场角
    fov_y: Float[Tensor, " batch"], # fov_y：垂直视场角
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


# 功能：使用 CUDA 加速的 splatting 方法进行渲染，得到每个像素的颜色值。
# 输入：
#   extrinsics: 相机外参矩阵 [B, 4, 4]
#   intrinsics: 相机内参矩阵 [B, 3, 3]
#   near: 近裁剪面距离 [B]
#   far: 远裁剪面距离 [B]
#   image_shape: 图像形状 (H, W)
#   background_color: 背景颜色 [B, 3]
#   gaussian_means: 高斯点均值 [B, G, 3]
#   gaussian_covariances: 高斯点协方差矩阵 [B, G, 3, 3]
#   gaussian_sh_coefficients: 高斯点球谐系数 [B, G, 3, d_sh] # d_sh：球谐系数数量，2阶的话d_sh=9，3阶的话d_sh=16
#   gaussian_opacities: 高斯点不透明度 [B, G]
# 输出：
#   color: 渲染得到的颜色图 [B, 3, H, W]
def render_cuda( # 此处的batch指的是图片的数量，gaussians指的是每张图片中高斯点的数量。
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],  # d_sh：球谐系数数量。2阶的话d_sh=9，3阶的话d_sh=16
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    # 在 scale-invariant（尺度不变）模式下，对相机外参和高斯体素进行缩放，以确保数值稳定性。
    # 这是因为在某些情况下，近裁剪面距离（near）可能非常小，导致计算中的数值不稳定。
    # 通过将相机外参和高斯体素缩放到一个更合适的范围，可以避免这种问题。
    if scale_invariant:
        scale = 1 / near                    # 计算一个缩放因子 scale，将 near 缩放到 1
        # 比如 near = 0.5，则 scale = 2.0，表示把场景放大 2 倍，使得近裁剪面变为单位长度。
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]  # 缩放相机外参中的平移向量(即相机位置)。
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2) # 缩放高斯体素的协方差矩阵。（协方差与长度的平方成正比，因此要乘 scale^2）
        gaussian_means = gaussian_means * scale[:, None, None] # 缩放高斯体的均值位置。
        near = near * scale # 缩放近裁剪面距离。
        far = far * scale   # 缩放远裁剪面距离。

    _, _, _, n = gaussian_sh_coefficients.shape # n 是球谐系数的数量，2阶的话 n=9，3阶的话 n=16。
    degree = isqrt(n) - 1  # 球谐阶数，n = (degree + 1)^2。比如 n=9 对应 degree=2， n=16 对应 degree=3。
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    # shs重排：(B, g, d_sh, 3)->(B, g, 3, d_sh)

    b, _, _ = extrinsics.shape
    h, w = image_shape
# 相机投影矩阵计算
    # 计算视场角（Field of View）的切线值
    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)  # 返回水平视场角fov_x 和 垂直视场角fov_y
    # 在透视投影中，近平面宽度一半 = near * tan(fov/2)，同理垂直方向。
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y) # 生成透视投影矩阵 [B, 4, 4]，用于把相机坐标系的点映射到裁剪空间
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i") # 转置投影矩阵，因为后续的矩阵乘法中需要使用转置的形式。
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")    # 转置外参矩阵(w2c)，得到视图矩阵（camera-to-world的逆矩阵，即world-to-camera矩阵）。
    full_projection = view_matrix @ projection_matrix         # 视图矩阵和投影矩阵相乘，得到完整的投影矩阵（从世界坐标系到裁剪空间的变换矩阵）。

    all_images = []     # 存放每个 batch 渲染出的图像。
    all_radii = []      # 存放每个 batch 渲染出的高斯半径（radii）
    for i in range(b):  # 循环的对每张图片进行渲染
        # Set up a tensor for the gradients of the screen-space means.
    # 作用：在训练或优化中，用于计算高斯中心在图像上的梯度（可用于几何优化）
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True) # 屏幕空间（2D 图像平面）高斯中心的梯度占位符。
        # 创建一个与高斯均值相同形状的张量 mean_gradients，并设置 requires_grad=True，以便在反向传播时计算其梯度。
        try:
            mean_gradients.retain_grad() # 对叶子节点保留梯度
        except Exception:
            pass

    # 构造高斯光栅化的设置对象
        settings = GaussianRasterizationSettings( # GaussianRasterizationSettings:封装了当前帧的所有相机参数和渲染配置。
            image_height=h,                 # 渲染图像尺寸
            image_width=w,
            tanfovx=tan_fov_x[i].item(),    # 水平/垂直视场角的正切，用于把 3D 点映射到屏幕空间
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],         # 背景颜色
            scale_modifier=1.0,             # 额外缩放因子（通常 1.0）
            viewmatrix=view_matrix[i],      # 视图矩阵（w2c）
            projmatrix=full_projection[i],  # 完整的投影矩阵（w2c @ projection）
            sh_degree=degree,               # 球谐函数的阶数
            campos=extrinsics[i, :3, 3],    # 相机位置（从外参矩阵中提取）
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
    # 实例化光栅化器，内部会根据 settings 设置屏幕空间映射、SH 光照等。
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3) # 取 3x3 对称矩阵的上三角索引。
        # 生成上三角矩阵的索引，用于提取协方差矩阵的独立元素（因为协方差矩阵是对称的，只需要存储上三角部分）。
    
    # 用光栅化器进行渲染，得到每个像素的颜色值和高斯半径。
        image, radii = rasterizer( # 调用光栅化器进行渲染，得到渲染图像和高斯半径。
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image) # 渲染后的 2D 图像 [H,W,3]
        all_radii.append(radii)  # 渲染后的高斯半径 [g, 2]（每个高斯在屏幕空间的半径）
    return torch.stack(all_images) # 堆叠所有 batch 的渲染结果，得到最终的渲染图像 [B, H, W, 3]


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

# 用cuda splatting 方法进行渲染，得到每个像素的颜色值和（可选的）深度图。
def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum( # 将高斯坐标变换到相机空间
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2] # 取 z 坐标作为“颜色”，因为我们关心的是深度信息。
    # 这个 fake_color 的形状是 [B, G]，每个高斯点对应一个深度值。

    # 模式选择：默认为 depth
    # "depth"：直接使用 z 作为颜色。
    # "disparity"：颜色 = 1/z（深度反比），常用于 Stereo。
    # "log"：颜色 = log(depth)，对深度范围变化大的场景更好，可增强近处细节。
    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color. （每个像素的颜色值对应于该像素的深度信息。）
    b, _ = fake_color.shape
    result = render_cuda( # 调用render_cuda渲染图片，这里面返回以深度作为颜色的渲染图片
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device), # 背景颜色设置为黑色，因为我们关心的是深度信息，背景的颜色不重要。
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3), # 球谐属性用fake_color代替。将 fake_color 从 [B, G] 扩展到 [B, G, 3]，使其可以作为颜色输入到 render_cuda 中。
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    
    return result.mean(dim=1)
