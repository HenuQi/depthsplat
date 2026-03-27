# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

# from vggt.models.aggregator import Aggregator
# from vggt.heads.camera_head import CameraHead
# from vggt.heads.dpt_head import DPTHead
# from vggt.heads.track_head import TrackHead
# ModuleNotFoundError: No module named 'vggt'，找不到，改成下面的

from ...vggt.models.aggregator import Aggregator
from ...vggt.heads.camera_head import CameraHead
from ...vggt.heads.dpt_head import DPTHead
from ...vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    # 继承 nn.Module → 这是标准 PyTorch 模型。
    # 继承 PyTorchModelHubMixin → 允许用 .from_pretrained() 从 HuggingFace 或自定义 hub 加载预训练权重。
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        # 5个模块初始化
        # Aggregator模块负责对图像处理然后输出token
        # CameraHead模块负责预测相机位姿, DPTHead模块负责预测 深度图 和 点图，
        # 还有一个可选的TrackHead模块，如果提供了需要跟踪的点那么它会输出跟踪点。
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim) # 输出 aggregated_tokens_list（tokens 序列，维度为2*embed_dim）和 patch_start_idx（每个 patch 的起始 index，用于重建）

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None # 说明送进去的 token 维度是dim_in=2 * embed_dim,
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None # 输出维度是4，预测 (x, y, z, w)
        
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid") if enable_depth else None
        # 为了把depth_conf预测的激活函数改成sigmoid，直接预测出来元素范围0~1的depth_conf，
        # 不过加载的vggt预训练权重是用expp1训练的，需要微调。
        # 原版如下:
        # self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None


    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # aggregator提取token，因为会同时输出camera相关的token 和 feature相关token
        # 需要patch_start_idx标记两者分界 
        # 输入： 
        #   images:(B, S, 3, H, W)
        # 输出：
        #   aggregated_tokens_list : 列表长度为24，每个元素的shape: (B, S, P, 2C)
        #       存的是每次 frame attention block 和 global attention block 输出的 token 拼接起来的结果
        #   patch_start_idx : int数，
        #       表示patch token 在 token 序列中开始的位置索引。每一帧的token顺序是: [ camera_token | register_tokens | patch_tokens ]，所以 patch_start_idx = 1 + num_register_tokens = 5
        aggregated_tokens_list, patch_start_idx = self.aggregator(images) # images:(S, 3, H, W)  or [B, S, 3, H, W],

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            # 计算相机位姿
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            # 计算深度图
            if self.depth_head is not None:
                # 我没用到vggt的forward(),这一步也要更新 深度图吗？
                # 注意，这里的 depth_head 不仅输出深度图和深度置信度，还输出了一个特征图（维度是 (B, S, C, H, W)），这个特征图是从 depth_head 内部提取的，用于后续的高斯参数回归。
                #原版：
                # depth, depth_conf = self.depth_head(
                #     aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                # )
                depth, depth_conf , feature_map= self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                ######新增 predictions["feature_map"] = feature_map

            # 计算点图
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # 可选的tracker模块用来估计给定点的轨迹
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

