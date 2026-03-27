# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# Aggregator模块对图像处理然后输出token
#  
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

# from vggt.layers import PatchEmbed
# from vggt.layers.block import Block
# from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
# from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from ...vggt.layers import PatchEmbed
from ...vggt.layers.block import Block
from ...vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ...vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block, # 
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100.0,  # 100 改成 100.0
        # VGGT本身没有启用启用了 jaxtyping + beartype，而DepthSPlat启用了 jaxtyping + beartype,所以接过来的时候会报错，暂时先改成 100.0 
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None # 如果 rope_freq > 0，就启用 2D Rotary Position Embedding
        self.position_getter = PositionGetter() if self.rope is not None else None # 如果启用了RoPE（rotary position embedding）旋转编码，就创建一个 PositionGetter，用来生成 2D patch 的空间坐标

        self.frame_blocks = nn.ModuleList( # 创建的frame_blocks，为一个长度为 depth 的列表，每一项都是一个 block_fn(...) 实例。
            [
                block_fn( # block_fn 是一个类，每一次循环都会调用：block_fn() 去生成一个 独立的 block 实例
                    dim=embed_dim,              # token 特征维度 C
                    num_heads=num_heads,        # attention head 数
                    mlp_ratio=mlp_ratio,        # MLP hidden dim = C * mlp_ratio
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size   # 24/1=24，表示frame attention和global attention交替的次数

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim)) # 创建一个可学习参数camera_token:(1,2,1,C=1024), 1:batch 维度的占位,2:每一个输入序列有 2 个camera token
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim)) # register_token:(1, 2, num_register_tokens=4, C=1024)

        # The patch tokens start after the camera and register tokens
        # patch token 在 token 序列中开始的位置索引。每一帧的token顺序是: [ camera_token | register_tokens | patch_tokens ]
        self.patch_start_idx = 1 + num_register_tokens # patch_start_idx = 1+4 = 5

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

# forward方法：对输入图像进行前向传播，输出每个 block 的中间结果（供后续 head 使用）和 patch_start_idx（供重建使用）
# 输入： 
#   images:(B, S, 3, H, W)
# 输出：
#   output_list : 列表长度为24，每个元素的shape: (B, S, P, 2C)
#       存的是每次 frame attention block 和 global attention block 输出的 token 拼接起来的结果
#   patch_start_idx : int数，
#       表示patch token 在 token 序列中开始的位置索引。每一帧的token顺序是: [ camera_token | register_tokens | patch_tokens ]，所以 patch_start_idx = 1 + num_register_tokens = 5
    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]: 
        # 输入: images:(B, S, 3, H, W), 且range [0, 1]：这个 tensor 里的所有数值都在 0 到 1 之间 
        # 输出: Tuple[List[torch.Tensor], int]。List[torch.Tensor]:(B, S, P, 2C), 2C:frame attention + global attention 的拼接
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        # 归一化,输入原本 ∈ [0,1], 现在变成类似 ImageNet 标准分布
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        # 把images，reshape 成 ViT 输入格式,然后用DINOv2 ViT的patch_embed来计算每张图的patch_tokens
        images = images.view(B * S, C_in, H, W) # images:(B,S,3,H,W) → (B*S,3,H,W)
        patch_tokens = self.patch_embed(images) # images:(B*S,3,H,W); patch_tokens:(B*S, P, C) P:每张图的patch数。C:每个patch的维度数
        # P=H//patch_size * W//patch_size = 518//14 * 518//14 = 37*37=1369, C=embed_dim=1024

        # 在 DINOv2 ViT 中, patch_embed输出的常常是 dict{"x_norm_patchtokens": Tensor, "x_norm_clstoken": Tensor, }
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"] # 取出其中真正的 patch tokens ：(B*S, P, C)

        _, P, C = patch_tokens.shape  # (B*S, P, C)：P:每张图的patch/token数。C:每个patch/token的维度数

    # 添加 camera_token 和 register_token ，
        # 先把它们扩展成适配多 batch、多帧的 token 序列
        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)  # self.camera_token:(1,2,1,C) -> camera_token:(B*S, 1, C)
        register_token = slice_expand_and_flatten(self.register_token, B, S) # self.register_token:(1,2,4,C) -> register_token(B*S, 4, C)

        # Concatenate special tokens with patch tokens
    # 把patch_tokens和camera_token和register_token,沿着第1维拼接起来
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1) # tokens:(B*S, 1+R+P, C) # 1+R+P:每张图的相机token数+寄存器token数+图的patch/token数

        # 因为每张图会划分成一块一块的patch，给每块 patch 添加 RoPE 位置编码（只给 patch token）
        # 因为只给 patch token进行位置编码（RoPE）:而camera_token和register_token又不需要这种块位置编码，所以跳过他们,
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device) # pos:(B*S, P, 2)

        if self.patch_start_idx > 0: # patch_start_idx=5 。如果使用了special tokens，就跳过它们编码
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1 # pos:(B*S, P, 2), 对整个张量 pos 的每个元素加 1，目的是区分特殊 token 和 patch token 的位置编码。加 1 后：特殊 token 坐标是 (1,1)，patch token 坐标从 (1,1) 开始 → 不会为 0
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype) # pos_special:(B*S,5,2)，其中元素全0
            # 两个to():把pos_special 转换到和images一致的device上，转换成和pos一致的数据类型
            pos = torch.cat([pos_special, pos], dim=1) # pos:(B*S, P, 2)  ->  (B*S, 1+R+P, 2)
            # 维度1+R+P的前5个元素为0，意味着这部分 token 的坐标都是 (0,0)

        # update P because we added special tokens
        _, P, C = tokens.shape # tokens:(B*S, 1+R+P, C)

        frame_idx = 0 # frame attention block 的索引
        global_idx = 0 # global attention block 的索引
        output_list = [] # 用来存储每次 block 输出的中间结果（后面会做 concat）

        # 重复交替执行frame_attention和global_attention就完事了
        for _ in range(self.aa_block_num):  # aa_block_num=24，表示frame attention 和 global attention 交替的次数（各24次）
            for attn_type in self.aa_order: # self.aa_order : ["frame","global"]
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention( # tokens:(B*S, 1+R+P, C) ->
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # 将每一次 frame attention block 和 global attention block 输出 token 拼接起来，供后续 head 使用
            # 最终列表 output_list 存起来，长度为24，每个元素的shape: (B, S, P, 2C)
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1) # frame + global = [B, S, P, C] + [B, S, P, C] →  concat_inter:(B, S, P, 2C)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

# 帧注意力处理
# 输入：
#   tokens:(B*S, P=1+R+P, C)，P:每张图的token数（包括特殊token和patch token），C:每个token的维度数
# 输出：
#   tokens:(B*S, P=1+R+P, C)，经过一个 frame attention block 处理后的 token
#   frame_idx:下一个要用的 frame attention block 的索引
#   intermediates:一个列表，保存了每次 frame attention block 输出的 token ，供后续 concat 使用。
#       列表中每个元素的形状都是 (B, S, P, C)，即把 (B*S, P, C) reshape 成 (B, S, P, C) 存下来，之后和 global attention 的输出拼接
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None): # tokens:(B*S, P=1+R+P, C), P=1+R+P
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C): # tokens 在经过 global attention 后是 (B, S*P, C)？？有时是 (B, S, P, C)
            tokens = tokens.view(B, S, P, C).view(B * S, P, C) # 保证tokens是(B*S, P, C), 每张图像的patch token就会分别被单独看成是一个 batch

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = [] # 保存中间结果

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size): # 默认 aa_block_size = 1，表示一次只用 1 个 frame block，就切换到global attention。如果设为 >1 → 连续用多个 frame attention block 再切到 global
            if self.training: # 如果当前模型处在训练模式。model.train() → self.training = True
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant) # 使用checkpoint梯度检查点机制。前向传播时不保存中间激活值，反向传播时重新计算这些激活，从而显著降低显存占用。
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos) # tokens(B*S,P+R+1,C)，pos:(B*S,P+R+1,2) -> tokens(B*S,P+R+1,C)
                # frame_blocks是长24的列表 nn.ModuleList，即内部是24层frame attention。 [frame_idx]列表取值取出单个block，(tokens, pos=pos)为这个block传入参数
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C)) # 把 (B*S, P, C) reshape 成(B, S, P, C)存下来，之后和 global attention 的输出拼接

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C) # 保证tokens是(B, S*P, C), 所有图的 patch token 都会被看成为同一个batch

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):  # 输入(1, 2, X, C) -> 输出(B·S, X, C)
    # 把形状为 (1, 2, X, C) 的特殊 token，扩展成适配多 batch、多帧的(B·S, X, C) token 序列，用于 Transformer 处理。
    # 
    # X = token 数量
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)   
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # query = Tensor(1, 1, X, C).expand(B, 1, X, C) ，表示第一帧的 相机/寄存器 token？
    #   token_tensor[:, 0:1, ...]:  (1, 2, X, C) -> (1, 1, X, C) 
    #   .expand(B, 1, X, C) :
    #       token_tensor.shape[2:]: (1, 2, X, C) -> (X, C)
    #       *token_tensor.shape[2:]: (X, C) -> X, C
    #       所以expand(B, 1, *token_tensor.shape[2:]) 实际上就是 .expand(B, 1, X, C)

    # Slice out the "other" tokens => shape (1, S-1, ...) 
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:]) 
    # others = Tensor(1, 1, X, C).expand(B, S-1, X, C) ，表示除了第一以外的其他 相机/寄存器 token？
    
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)  # (B, 1, X, C)+(B, S-1, X, C) -> (B, S, X, C)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:]) # combined:(B, S, X, C) -> (B*S, X, C) 
    return combined
