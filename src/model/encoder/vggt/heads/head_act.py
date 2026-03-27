# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

# 从vggt.py中学习，定义一些激活函数，用于处理VGGT的输出，得到最终的深度图和置信度图。
def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")

# 定义一个函数 activate_head()，用于处理DPTHead的输出，得到最终的深度图和置信度图。
# 输入：
#   out = (B*S, 2, H, W) 
#   activation="exp"
#   conf_activation="expp1"
# 输出：
#   pts3d: (B*S, 3, H, W) 预测的3D点坐标
#   conf_out: (B*S, H, W) 预测的置信度图
def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected
    # fmap: (B*S, H, W, 2) 2个通道分别是：预测深度，以及预测的深度置信度。

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]  #  xyz: (B*S, H, W, 1)    # 3个参数是 xyz
    conf = fmap[:, :, :, :-1] #  conf: (B*S, H, W, 1)      # 1个参数是 置信度
    # 应该是写错了？？？？本来是
    # conf = fmap[:, :, :, -1]  #  conf: (B*S, H, W)      # 1个参数是 置信度
    # 我改成：
    # conf = fmap[:, :, :, :-1] #  conf: (B*S, H, W, 1)      # 1个参数是 置信度

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp": # depth_head默认对xyz使用的 exp 激活函数
        pts3d = torch.exp(xyz) # pts3d: (B*S, H, W, 3) 预测的3D点坐标，值域是(0, +inf)，因为 exp(x) > 0 对于所有实数x都成立。
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":  # 点头_head默认对xyz使用的 inv_log激活函数
        pts3d = inverse_log_transform(xyz) # pts3d: (B*S, H, W, 3) 经过逆对数变换后的3D点坐标
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1": # depth_head默认对置信度使用expp1激活函数
        conf_out = 1 + conf.exp() # conf_out: (B*S, H, W) 预测的置信度图，值域是(1, +inf)，因为 expp1(x) = 1 + exp(x)，当x趋近于负无穷时，expp1(x)趋近于1，当x趋近于正无穷时，expp1(x)趋近于正无穷。
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out
    # pts3d: (B*S, H, W, 3) 预测的3D点坐标，值域是(0, +inf)，因为 exp(x) > 0 对于所有实数x都成立。
    # conf_out: (B*S, H, W) 预测的置信度图，值域是(1, +inf)，因为 expp1(x) = 1 + exp(x)，当x趋近于负无穷时，expp1(x)趋近于1，当x趋近于正无穷时，expp1(x)趋近于正无穷。

# 输出是经过非线性增强后的 位移场,
# 输入：xyz: (B*S, H, W, 3)
# 输出： (B*S, H, W, 3) 经过逆对数变换后的3D点坐标
def inverse_log_transform(y): 
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

