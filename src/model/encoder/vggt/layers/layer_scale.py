# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module): # 定义了一个 PyTorch 模块，用于对特征做一个可学习的逐通道缩放（scale）
    # LayerScale 的数学意义:给每个通道一个可学习的缩放因子 γ，每个通道维度 i 都乘上一个可学习缩放系数 γᵢ
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace # 是否使用 inplace 操作（原地修改,不创建新变量）
        self.gamma = nn.Parameter(init_values * torch.ones(dim)) # gamma:(C)实际上是一个长度为 C 的一维张量，用nn.Parameter()把它注册为可学习参数。
        # gamma 不是真的变成了一个新 Tensor (B,N,C)，
        # 而是通过 broadcast 广播机制在运算时“虚拟扩展”到 (B,N,C)。

    def forward(self, x: Tensor) -> Tensor:  # x:(B,N,C) {B:batch, N:token数[在VGGT中此时的N=1+4+P], C:token维度} gamma:(C)-> (B,N,C) {输出的C中每一个元素都乘上了一个可学习缩放系数 γᵢ}
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    # 如果 inplace  =  True，则原地修改 
    # x.mul_(gamma) ： 直接在原来的 x 上乘，不新建 Tensor
    # x * self.gamma ： 分配一个新 Tensor，原来的 x 不变 
