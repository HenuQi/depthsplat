# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,  # dim_in = 2C（帧注意力+全局注意力的特征维度拼起来），C=1024
        patch_size: int = 14,
        output_dim: int = 4,            # depth_head传入了 output_dim = 2
        activation: str = "inv_log",    # depth_head传入了 activation = exp
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,     # 如果 feature_only=True，则返回特征图而不是最终的预测结果。这对于需要使用 DPT 提取的特征进行其他任务（如点云重建）而不是直接进行深度预测的情况非常有用。
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList( # in_channels = dim_in = 2C，C=1024， out_channels = [256, 512, 1024, 1024] 
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        # 实际上创建了四层 Conv2d 卷积，如果调用self.projects[1]就只执行 Conv2d(2C→512, kernel=1) → 输出通道数512。
        # self.projects[0]: Conv2d(2C→256, kernel=1) → 输出通道数256
        # self.projects[1]: Conv2d(2C→512, kernel=1) → 输出通道数512
        # self.projects[2]: Conv2d(2C→1024, kernel=1) → 输出通道数1024
        # self.projects[3]: Conv2d(2C→1024, kernel=1) → 输出通道数1024

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList( # in_channels 和 out_channels 都是 out_channels = [256, 512, 1024, 1024]，进行上采样，
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        # resize_layers实际上也是4个层分别执行，而不是一次执行全部
        # 比如调用self.resize_layers[2]就只执行nn.Identity()
        # self.resize_layers[0]: ConvTranspose2d(256→256, kernel=4, stride=4) → 通道数不变，上采样4倍
        #       意思是：如果x:(BV, 256, H/14, W/14)，则经过self.resize_layers[0]这一层之后变成：x:(BV, 256, H/14*4, W/14*4)
        # self.resize_layers[1]: ConvTranspose2d(512→512, kernel=2, stride=2) → 通道数不变，上采样2倍
        # self.resize_layers[2]: Identity() → 不改变尺寸
        # self.resize_layers[3]: Conv2d(1024→1024, kernel=3, stride=2, padding=1) → 通道数不变，下采样2倍

        self.scratch = _make_scratch(out_channels, features, expand=False) 
        # out_channels: [256, 512, 1024, 1024]，features=256，expand=False
        # 返回这个包含了多个卷积层的 scratch 模块，这些卷积层会在 DPT 的多尺度融合过程中被分别调用，
        # 用于把来自不同 Transformer 层的 feature 统一成“可融合的通道结构”，即输出通道数均为256，以便后续的融合块进行特征融合和预测。

        # 定义了模型中 解码器路径（Decoder Path）的多级融合模块，
        # 核心目标是：从编码器输出的多尺度特征中，构建高分辨率、语义丰富的最终输出特征图。
        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None # 模型不使用“stem transpose”操作（如转置卷积上采样）
        self.scratch.refinenet1 = _make_fusion_block(features) 
        # 构建 第一个融合块（RefineNet Block），用于处理 最低分辨率、最高语义的特征图

        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        # 构建 第四个融合块，用于把最低分辨率、最高语义的特征图（layer_4_rn）上采样到和下一层特征图（layer_3_rn）相同的分辨率，然后进行融合，得到新的特征图，
        # 

        head_features_1 = features
        head_features_2 = 32

        # 定义output_conv1
        if feature_only: # 如果只需要特征图而不是最终的预测结果(深度图)
            # output_conv1 是一个简单的卷积层，用于把融合后的特征图的通道数统一成 head_features_1（默认为256），以便直接输出特征图。
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            # 如果需要最终的预测结果（如深度图），
            # 则 output_conv1 是一个卷积层，用于把融合后的特征图的通道数从 head_features_1（默认为256）降维到 head_features_1 // 2（默认为128），
            self.scratch.output_conv1 = nn.Conv2d( # head_features_1 = 256, head_features_1 // 2 = 128
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2 # conv2_in_channels = 256/2 = 128

            # output_conv2 是一个卷积层，用于把上采样后的特征图的通道数从 conv2_in_channels（默认为128）降维到 output_dim（depth_head默认为2），
            self.scratch.output_conv2 = nn.Sequential( # conv2_in_channels=128，head_features_2 = 32, output_dim = 2
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

# 前向函数
# 输入：
#   aggregated_tokens_list: list of (B, V, P, 2C)，从VGGT的aggregator提取的特征tokens列表
#   images: [B, S, 3, H, W]，输入图像，适配VGGT输入格式（已经被预处理成518*518）
#   patch_start_idx: int，patch token在token序列中的起始索引，用于切分掉camera/register token
# 输出：
#   preds: (B, S, H, W, 1)
#   conf: (B, S, H, W, 1)
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # 加上了：Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        # 原版：-> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        # 是否需要 chunk。DPT head 很吃显存，一次性 forward所有图片可能会显存爆掉，所以可以通过 frames_chunk_size 参数指定每次 forward 多少帧图像，以控制显存使用。
        # tokens = (B, S, P, C)
        # DPT 会：reshape 成 feature map， 上采样到 H×W， 多层融合
        if frames_chunk_size is None or frames_chunk_size >= S: # 如果不指定 chunk 大小，或者 chunk 大小大于序列长度 S，则一次性处理所有帧
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        # 初始化容器，用于存储每个 chunk 的输出结果，最后再拼接起来返回。
        all_preds = []
        all_conf = []
        all_feature = []  ########## 新增，接受特征图的输出

        for frames_start_idx in range(0, S, frames_chunk_size): # 按帧切块处理，frames_start_idx 是当前 chunk 的起始帧索引，frames_end_idx 是当前 chunk 的结束帧索引（不包含）
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S) # 计算chunk范围，确保结束索引不超过图像序列长度 S

            # Process batch of frames
            if self.feature_only: # 如果只需要特征图而不是最终的预测结果(深度图)，则直接返回特征图，不经过最后的几层和激活头。
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else: # 否则，正常返回预测结果和置信度。
                chunk_preds, chunk_conf, chunk_feature = self._forward_impl( 
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)   #  (B, S, H, W, 1)
                all_conf.append(chunk_conf)     #  (B, S, H, W, 1)
                all_feature.append(chunk_feature) ############ 新增，接收特征图返回 (B, S, C, H, W)

        # Concatenate results along the sequence dimension
        if self.feature_only: # 如果只需要特征图，直接拼接特征图返回。
            return torch.cat(all_preds, dim=1)
        else: # 否则，拼接多个块的结果，把预测的深度图和置信度图一起返回。
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1), torch.cat(all_feature, dim=1)  ########## 新增，拼接特征图返回




# 具体计算特征图 / 深度图+置信度的函数。
# 输入：
#   aggregated_tokens_list: list of (B, V, P, 2C)，从VGGT的aggregator提取的特征tokens列表
#   images: [B, S, 3, H, W]，输入图像，适配VGGT输入格式（已经被预处理成518*518）
#   patch_start_idx: int，patch token在token序列中的起始索引，用于切分掉camera/register token
# 输出：
#   如果 feature_only=True，则返回特征图，shape为[B, S, C, H, W]
#   否则，返回预测的深度图和置信度图，shape都是[B, S, 1, H, W]
    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:  
        # 加上了：Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        # 原版：-> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None: # 如果指定了帧范围，则切片出当前 chunk 的图像进行处理。
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        # patch_size = 14， patch_h = patch_w=37 = 518//14=37， 
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # 多层 Transformer 特征提取
        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx: # intermediate_layer_idx : [4, 11, 17, 23]
            # 分别用 aggregated_tokens_list 中第 4, 11, 17, 23 层提取的特征计算特征图
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:] # 去掉非 patch token的部分（即去掉camera/register token）

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None: # chunk 的图片和token对齐，如果指定了帧范围，则切片出当前 chunk 的 token 进行处理。
                x = x[:, frames_start_idx:frames_end_idx]

            # 此时的 x 是 patch token 的部分，shape是 (B, S, P, 2C)，需要把它变成 feature map 的形式，才能输入到后续的卷积层进行处理。
            # token → feature map（核心变换）
            x = x.reshape(B * S, -1, x.shape[-1]) # x: (B, S, P, C) → (B*S, P, C)，把 batch 和 sequence 维度合并

            x = self.norm(x) # 对 token 进行 layer normalization，保持数值稳定性

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)) # x: (B*S, P, C) → (B*S, C, patch_h, patch_w)
            # 把 token 维度变成 channel 维度.
            # patch 维度变成 feature map 的空间维度. patch_h = H/14 
            # 此时的 x shape为 (B*S, C, H/14 , W/14 )

            x = self.projects[dpt_idx](x) # 通过 1x1 卷积把 token 的维度投影到指定的通道数 out_channels：[256, 512, 1024, 1024]，得到新的 feature map，
            # 第一次循环，此时x的shape是 (B*S, 256, patch_h, patch_w)，patch_h = H/14 

            if self.pos_embed:  # 位置编码（可选）,给 feature map 加 spatial awareness.
                # 注意：ViT token → map 位置信息会弱化，所以 DPT 要补位置编码
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x) # 上采样 feature map 到更高的分辨率，接近输入图像的分辨率，以便后续融合和预测。
             # 第一次循环，此时 x 的 shape 是 (B*S, 256, patch_h*4, patch_w*4)

            out.append(x) # 收集多层特征
            # out 是一个列表，长度为intermediate_layer_idx : [4, 11, 17, 23]的长度 4， 
            # 4个元素分别是用aggregated_tokens_list中第4, 11, 17, 23层提取的特征，经过投影和上采样后的 feature map，
            # 每个元素是一个 feature map，shape分别是 ： (patch_h = H/14 ，patch_w = W/14 )
            # out = [
            #     (B*S, 256, patch_h*4, patch_w*4)
            #     (B*S, 512, patch_h*2, patch_w*2)
            #     (B*S, 1024, patch_h, patch_w)
            #     (B*S, 1024, patch_h/2, patch_w/2)
            #     ]
            dpt_idx += 1

        # Fuse features from multiple layers.
    # DPT 多尺度融合（核心）
        # 只输出特征图：
        # 同时输出
        out = self.scratch_forward(out)  # 类似 U-Net：multi-layer features → 上采样 + skip connection → 融合
        # out: 融合了 [4, 11, 17, 23] 四个特征得到的特征图
        # out：(B*S, 256/2=128, patch_h*4, patch_w*4)

        # Interpolate fused output to match target image resolution.
    # 把融合的特征上采样到目标分辨率
        out = custom_interpolate(  # 上采样插值，out: (B*S, 256/2=128, patch_h*4, patch_w*4) -> (B*S, 128, H, W)    注意：patch_h = H/14
            out, # (B*S, 256/2=128, patch_h*4, patch_w*4)
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            # 上面一行实际上是恢复到原分辨率 size：(H, W)
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed: # 再次加位置编码，out ：(B*S, 256/2=128, H, W)
            out = self._apply_pos_embed(out, W, H)
    # 如果只需要特征图，直接返回特征图，不经过最后的几层和激活头。
        if self.feature_only: 
            return out.view(B, S, *out.shape[1:])
        # 此时的特征图是融合了VGGT[4, 11, 17, 23]层的特征，并恢复到原分辨率后的特征图，shape是 (B, S, C=128, H, W)  # 注意，C不一定是128，
        # 如果 feature_only=True，则 output_conv1 的输出通道数是 head_features_1 = 256，所以 C=256；
        # 如果 feature_only=False，则 output_conv1 的输出通道数是 head_features_1 // 2 = 128，所以 C=128。

#########################################新增
        # 设计成在输出深度图的同时，也输出特征图
        feature_map = out.view(B, S, *out.shape[1:]) # feature_map: (B, S, C, H, W)
#########################################

    # 如果不是只需要特征图，则继续通过最后的卷积层和激活头得到最终的预测结果（深度图和置信度图）。
        out = self.scratch.output_conv2(out) # out ：(B*S, 256/2=128, H, W) -> (B*S, 2, H, W) 
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation) # activation=exp, conf_activation=expp1
        # preds: (B*S, H, W, 1) 预测的3D点坐标，值域是(0, +inf)，因为 exp(x) > 0 对于所有实数x都成立。
        # conf: (B*S, H, W, 1) 预测的置信度图，值域是(1, +inf)，

        # 把 batch 和 sequence 维度分开
        preds = preds.view(B, S, *preds.shape[1:]) # preds: (B*S, H, W, 1) -> (B, S, H, W, 1)
        conf = conf.view(B, S, *conf.shape[1:]) # conf: (B*S, H, W, 1) -> (B, S, H, W, 1)
        return preds, conf, feature_map  ################# 新增了 feature_map:(B, S, C, H, W),原版如下：
        # return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    # DPT 多尺度融合（核心）：类似 U-Net：multi-layer features → 上采样 + skip connection → 融合
    # 输入： [4, 11, 17, 23]层提取的特征，经过投影和上采样后的 feature map
    # out = [
    #     (B*S, 256, patch_h*4, patch_w*4),
    #     (B*S, 512, patch_h*2, patch_w*2),
    #     (B*S, 1024, patch_h, patch_w),
    #     (B*S, 1024, patch_h/2, patch_w/2)
    #     ]
    # 输出：
    # out = (B*S, features, patch_h*4, patch_w*4)，融合了多层特征的 feature map，分辨率接近输入图像的分辨率。

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features 
        # layer_1 : (B*S, 256, patch_h*4, patch_w*4)
        # layer_2 : (B*S, 512, patch_h*2, patch_w*2)
        # layer_3 : (B*S, 1024, patch_h, patch_w)
        # layer_4 : (B*S, 1024, patch_h/2, patch_w/2)

    # 四层特征进行通道统一，输出通道数都是 features=256，尺寸都不变。
        layer_1_rn = self.scratch.layer1_rn(layer_1) # 卷积
        # scratch.layer1_rn: Conv2d(256→256, kernel=3) → 输出通道数256, 尺寸不变
        # 输入：(B*S, 256, patch_h*4, patch_w*4) 输出：(B*S, 256, patch_h*4, patch_w*4)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

    # 在通道统一后，进行多层融合，
    # 融合的方式是：先把最低分辨率、最高语义的特征图（layer_4_rn）上采样到和下一层特征图（layer_3_rn）相同的分辨率，然后进行融合，得到新的特征图，
    # 再把这个新的特征图上采样到和下一层特征图（layer_2_rn）相同的分辨率，再进行融合，以此类推，直到最后得到融合后的特征图。
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]) 
            # layer_4_rn:(B*S, 256, patch_h/2, patch_w/2)  size:(patch_h, patch_w) 
            # 输出：layer_4_rn上采样后和layer_3_rn尺寸相同，然后融合，得到新的特征图 out：(B*S, 256, patch_h, patch_w)
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        # 四层特征融合后的特征图 out 的 shape 是 (B*S, features=256, patch_h*4, patch_w*4)。

        out = self.scratch.output_conv1(out)
        # 经过output_conv1卷积层之后的out：(B*S, 256/2=128, patch_h*4, patch_w*4)
        return out
    


################################################################################
# Modules
################################################################################


# 构建融合块，核心是 FeatureFusionBlock，负责把来自不同层的特征融合在一起，形成更丰富的特征表示，以便后续的预测。
# 把最低分辨率、最高语义的特征图（layer_4_rn）上采样到和下一层特征图（layer_3_rn）相同的分辨率，然后进行融合
def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )

# 把来自不同 Transformer 层的 feature 统一成“可融合的通道结构”
# 输入：
def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    # in_shape = [256, 512, 1024, 1024] , out_shape = 256
    scratch = nn.Module() # 创建容器：scratch 是一个空的 nn.Module，用于动态添加卷积层，
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand: # 默认为 false，如果 expand=True，则把输出通道数翻倍，以便后续的融合块有更多的特征维度可以使用。
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d( # 定义layer1_rn层是一个 可学习的 2D 卷积操作
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # scratch.layer1_rn: Conv2d(256→256, kernel=3) → 输出通道数256, 尺寸不变
    # 输入：(B*S, 256, patch_h*4, patch_w*4) 输出：(B*S, 256, patch_h*4, patch_w*4)，只是

    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # scratch.layer2_rn: Conv2d(512→256, kernel=3) → 输出通道数256, 尺寸不变

    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    # scratch.layer3_rn: Conv2d(1024→256, kernel=3) → 输出通道数256, 尺寸不变

    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch
    # 返回这个包含了多个卷积层的 scratch 模块，这些卷积层会在 DPT 的多尺度融合过程中被分别调用，
    # 用于把来自不同 Transformer 层的 feature 统一成“可融合的通道结构”，即输出通道数均为256，以便后续的融合块进行特征融合和预测。


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size


    # 特征融合块的前向函数，把低分辨率的特征图（如：layer_4_rn）上采样到和下一层特征图（如:layer_3_rn）相同的分辨率，然后进行融合
    # # 
            
    # 输入：
    #   如layer_4_rn的特征:(B*S, 256, patch_h/2, patch_w/2)  
    #   layer_3_rn的特征的size:(patch_h, patch_w) 
    # 输出：
    #   layer_4_rn上采样后和layer_3_rn尺寸相同，然后融合，
    #   得到新的特征图 out：(B*S, 256, patch_h, patch_w)
    # 
    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual: # 默认为True残差连接融合
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2.0}  #### 从"scale_factor": 2改成 "scale_factor": 2.0
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


# 
def custom_interpolate( # 
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
