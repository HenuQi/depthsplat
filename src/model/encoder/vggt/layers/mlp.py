# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module): # Vision Transformer / VGGT / DINO 里的标准 MLP（FFN, Feed-Forward Network）模块
    def __init__(
        self,
        in_features: int,   # 输入通道维度 C
        hidden_features: Optional[int] = None,  # 中间隐藏层维度，ViT 中通常：hidden_features = 4 * in_features
        out_features: Optional[int] = None, # 输出通道维度
        act_layer: Callable[..., nn.Module] = nn.GELU, # 激活函数类型
        drop: float = 0.0, # Dropout 概率
        bias: bool = True, # Linear 是否带 bias
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias) # 第一层全连接, 用于通道升维 / 特征扩展
        self.act = act_layer() # 激活函数，默认为 ViT 的标配nn.GELU()，比 ReLU 更平滑，
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias) # 第二层全连接，用于把维度压回原通道 C
        self.drop = nn.Dropout(drop) # Dropout 用于正则化，防止过拟合x:

    def forward(self, x: Tensor) -> Tensor:  # x:(B, N, C) -> (B, N, C) 变化是：每个 token 的 C 维向量被非线性地重编码了一次。输入的 C 是“原始 token 特征”，输出的 C 是“经过通道混合 + 非线性变换后的新特征表示”。
        x = self.fc1(x) # (B, N, C) → (B, N, hidden)
        x = self.act(x) 
        x = self.drop(x)
        x = self.fc2(x) # (B, N, hidden) → (B, N, C)
        x = self.drop(x)
        return x
# MLP的整体结构：两层全连接 + 激活 + Dropout
# x → Linear(in → hidden) → GELU → Dropout → Linear(hidden → out) → Dropout → 输出（方便 residual 相加）
# 输入输出维度一致，用在残差结构里。
