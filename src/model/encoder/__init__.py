from typing import Optional

from .encoder import Encoder
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_depthsplat import EncoderVisualizerDepthSplat

from .encoder_vggt_depthsplat import EncoderVGGTDepthSplat, EncoderVGGTDepthSplatCfg

# EncoderVGGTDepthSplat的可视化器还没有改好
ENCODERS = {
    # "depthsplat": (EncoderDepthSplat, EncoderVisualizerDepthSplat),
    "vggtdepthsplat": (EncoderVGGTDepthSplat, EncoderVisualizerDepthSplat),  # 这里暂时先用 EncoderVisualizerDepthSplat 来替代，后续可以替换成基于 VGGT 的可视化器
}

# EncoderCfg = EncoderDepthSplatCfg 
EncoderCfg = EncoderDepthSplatCfg | EncoderVGGTDepthSplatCfg # 联合类型，表示 EncoderCfg 可以是 EncoderDepthSplatCfg 或 EncoderVGGTDepthSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
