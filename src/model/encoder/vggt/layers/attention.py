# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,  # rope默认为None。 VGGT实际使用时，传入RotaryPositionEmbedding2D类实例
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads #  注意力头 head 的数量
        self.head_dim = dim // num_heads #  每个 head 分到的通道维度大小
        self.scale = self.head_dim**-0.5 # 取倒数再开平方
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 一次性算出 Q, K, V（拼在一起）
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity() # 是否对 Q/K 做 LayerNorm（按 head_dim）
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope # 可选 RoPE 位置编码模块

    def forward(self, x: Tensor, pos=None) -> Tensor: # x:(B,N,C) {B:batch, N:token数[在VGGT中此时的N=1+4+P], C:token维度}-> (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # x:(B,N,C) -> (B,N,3C) -> (B,N,3,num_heads,head_dim) -> (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0) # qkv:(3, B, num_heads, N, head_dim)  ->  q/k/v:(B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k) # 可选 QK 归一化

        if self.rope is not None: # 加 RoPE（旋转位置编码）：在 Q/K 上注入空间位置信息，pos 通常是 (B, N, 2)，2这个维度上表示 (y,x)
            q = self.rope(q, pos)  #  传入RotaryPositionEmbedding2D类实例
            k = self.rope(k, pos)

        if self.fused_attn: # 用 PyTorch fused 实现计算注意力
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else: # 手写 attention
            q = q * self.scale # q:(B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)     self.scale = head_dim ** -0.5 = 1 / sqrt(head_dim)
            attn = q @ k.transpose(-2, -1) #  q:(B, num_heads, N, head_dim) 和 k转置:(B, num_heads, head_dim, N)，做 @ 矩阵乘法。-> attn:(B, num_heads, N, N)
            attn = attn.softmax(dim=-1) # 归一化最后一维
            attn = self.attn_drop(attn) #  dropout
            x = attn @ v # attn:(B, num_heads, N, N) , v:(B, num_heads, N, head_dim) @运算后 ->  x:(B, num_heads, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, C) # 合并 attention heads。 x:(B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, num_heads*head_dim)
        x = self.proj(x) # x:(B, N, num_heads*head_dim  = C) -> (B, N, num_heads*head_dim) :融合不同 attention head 的信息
        x = self.proj_drop(x)  
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
