import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.encoder.vggt.models.vggt import VGGT
from .model.encoder.vggt.run_vggt import preprocess_images_for_vggt, run_Aggregator





def main():

    # 1. 加载图片
    

    # 2. 加载模型
    # 创建模型，所有输出头都不要，只要aggregator
    vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)

    #-------------------------------------------------------------------------------（先不用，因为在main.py中配置了加载权重）
    # # 改成本地加载权重
    ckpt_url = "pretrained/vggt_model.pt"
    checkpoint = torch.load(ckpt_url, map_location="cpu")
    vggt.load_state_dict(checkpoint, strict=False)
    # load_state_dict()：把权重加载到模型中
    #   实际上就是state_dict[key]  →  model.parameter[key]，例如：
    #   "encoder.conv1.weight" → model.encoder.conv1.weight
    #   注意：key 必须完全匹配，shape 必须一致
    # ---------------------------------------------------------------------------------







