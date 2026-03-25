from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead

# 一、定义编码器配置类EncoderDepthSplatCfg, 在config/model/encoder/depthsplat.yaml中配置
@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int                                   # 配置中为 1，表示每个像素预测一个高斯球
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool                     # 默认为True，表示在GaussianAdapter中是否使用输入图像作为球谐函数的输入特征。这可以帮助模型更好地学习颜色信息与高斯参数之间的关系，从而提升最终的渲染质量。
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool   # 默认为True, 
    return_depth: bool                   # 默认为True

    # only depth   默认为false，如果为true，则只训练深度预测部分，不训练高斯参数回归部分。这对于预训练深度预测模块或者在某些阶段专注于提升深度预测性能可能有帮助。
    train_depth_only: bool   

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int

# 二、定义编码器类EncoderDepthSplat
class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
# 1. 定义初始化函数__init__()，根据配置初始化DepthPredictorMultiView、DPTHead、GaussianAdapter等模块。 
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)
# 1.1 初始化MultiViewUniMatch（depth_predictor），用于从多视图特征中预测深度图/raw Gaussian 参数。
        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }
# 1.2 初始化DPTHead（feature_upsampler），用于将多视图特征上采样到原始分辨率，以便后续的高斯参数回归。
        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
# 1.3 初始化GaussianAdapter（gaussian_adapter），用于将深度图和特征转换为高斯参数。
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels    # 
        channels = self.cfg.gaussian_regressor_channels         # 默认为 64 
    # 高斯回归头 gaussian_regressor
        # conv regressor
        modules = [ # 简单的卷积回归头
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)


        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1 
        # 总计为: 37  。2 个 offset, 1 个 opacity，34 个 d_in
        # d_in = 7 + 3*9 = 34。分别是 位置3，深度1，可信度1，二阶球谐系数数量(rgb三通道) 3*9=27

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
    # 高斯预测头 gaussian_head
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])
# 2. 定义前向传播函数forward()，实现从输入图像和相机参数到高斯参数的完整转换流程，包括深度预测、特征上采样、高斯参数回归等步骤。
    # 输入: "context": {                                                 # B指的是batch数，V指的是context view 的数量
    #     "extrinsics": extrinsics[context_indices],                          # (B,V,4,4)
    #     "intrinsics": intrinsics[context_indices],                          # (B,V,3,3)
    #     "image": context_images,                                            # (B,V,3,H,W)
    #     "near": self.get_bound("near", len(context_indices)) / nf_scale,    # (B,V)
    #     "far": self.get_bound("far", len(context_indices)) / nf_scale,      # (B,V)
    #     "index": context_indices,                                           # (B,V)
    # },
    # 输出：
    #       Gaussians(means, covariances, harmonics, opacities, scales, rotations)
    #           高斯位置：means:     [B, V*HW*1*1, 3]
    #           协方差：covariances: [B, V*HW*1*1, 3, 3]
    #           球谐系数：harmonics: [B, V*HW*1*1, 3, 9]
    #           不透明度：opacities: [B, V*HW*1, 1]
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3: # 如果视图数 > 3，就只做 局部视图匹配（减少计算量）
            with torch.no_grad(): #
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)] # cameras_dist_index用于为每个相机选最近的 K 个相机
        else: # 视图 <=2 时直接全部匹配
            cameras_dist_index = None
# 2.1 MultiViewUniMatch模块进行深度预测（depth_predictor）：
# 功能包括 CNN特征提取，Transformer特征融合，DINOv2单目特征提取，深度预测
        # depth prediction
        # 输入： 
        #   context image:(B,V,3,H,W), 
        #   intrinsics:(B,V,3,3), 
        #   extrinsics:(B,V,4,4), 
        #   near:(B,V), 
        #   far:(B,V)
        # 输出：results_dict：{
        #           "features_cnn_all_scales": features_list_cnn_all_scales,     # CNN提取的全部特征图列表 。                            每个元素:(B*V, C=128, H', W')，其中 H' 和 W' 是对应分辨率的大小，如 H/8,W/8 或 H/4,W/4 等。
        #           "features_cnn": features_list_cnn,                           # CNN提取的只包含1/8分辨率特征图 的列表。                每个元素:(B*V, C=128, H/8, W/8)。
        #           "features_mv": features_list_mv,                             # CNN+Transformer 提取融合的2D特征图。                  每个元素:(B*V, C=128, H/8, W/8)。 
        #           "features_mono_intermediate": mono_intermediate_features,    # DINOv2 的中间层特征图列表，包含了指定block的特征图。   每个元素:(B*V, C'=384, H/8, W/8)，其中C'是对应block的特征维度，如 ViT-S:384，ViT-B:768，ViT-L:1024。
        #           "features_mono": features_list_mono,                         # DINOv2 的最后一个block的特征图（分辨率为1/8）。        每个元素:(B*V, C'=384, H/8, W/8)，其中C'是对应block的特征维度，如 ViT-S:384，ViT-B:768，ViT-L:1024。
        #           "depth_preds": depth_preds,                                  # 深度预测的 深度图depth列表（已经从逆深度转换为深度）。  每个元素:(B, V, H, W)。
        #           "match_probs": match_probs,                                  # 匹配概率图列表。                                      每个元素:(BV, D, H/8, W/8)，
        #       }
        results_dict = self.depth_predictor(
            context["image"],                       # (B,V,3,H,W)
            attn_splits_list=[2],             # 多视图注意力的分割数列表，默认为[2]，表示在多视图注意力中将视图分成两组进行计算，以减少内存占用和计算量。
            min_depth=1. / context["far"],          # (B,V)
            max_depth=1. / context["near"],         # (B,V)
            intrinsics=context["intrinsics"],       # (B,V,3,3)
            extrinsics=context["extrinsics"],       # (B,V,4,4)
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only: # （默认不走）如果只训练深度预测部分，就直接返回深度图，不进行后续的高斯参数回归。
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }
# 2.2 DPTHead模块进行特征上采样（feature_upsampler）：把1/8分辨率的特征图上采样到原始分辨率，以便后续的高斯参数回归。
# 输入：
#       features_mono_intermediate：    [(B*V, C'=384, H/8, W/8), ]   # DINOv2的中间层特征图,一共4个内容
#       features_cnn_all_scales:        [(B*V, C=128, H/8, W/8), 1/4, 1/2...] # CNN提取的全部特征图
#       features_mv:                    [(B*V, C=128, H/8, W/8)]  # CNN+Transformer融合的特征图
#输出：
#       上采样后的特征图 features         [BV, C, H, W]，其中C是上采样后特征图的通道数，H和W是原始图像的分辨率。
#
        # features [BV, C, H, W] 
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )
    # 求深度可信度match_prob:(BV, 1, H, W)
        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1] # match_prob:[BV, D, H, W] 最后一层（最高分辨率）的概率,
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[ # 沿 D 维度取最大值。如果最大值很大接近 1，说明匹配的准确。# max probability≈ 匹配置信度
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate( # nearest插值，恢复到原分辨率
            match_prob, size=depth.shape[-2:], mode='nearest')
        
        # 拼接：原始图片，深度图，深度可信度，融合特征 得到 concat:(BV, C_total, H, W)，
        # C_total= 3 + 1 + 1 + feature_upsampler_channels（取决于不同的DiNOv2模型，分别是64，96，128） 
        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)
# 2.3 gaussian_regressor和gaussian_head进行初始高斯参数预测：
# 2.3.1 gaussian_regressor 进行特征融合和转换, 把特征变成适合gaussians预测的表示，再由gaussian_head做参数预测。
        # gaussian_regressor() 是用于高斯参数回归的卷积头
        # 
        # 输入：融合的特征图 concat:(BV, C_total, H, W])
        # 输出：回归的特征图 out:(BV, 64, H, W)      
        #   其中64是gaussian_regressor_channels的默认值。
        #   这个特征图将被用于后续的高斯参数预测。
        out = self.gaussian_regressor(concat)

        # 拼接：回归特征图，原始图片，融合特征，深度可信度，得到一个列表 concat
        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1) # 拼接最终输入gaussian_head的特征图
        # 将concat列表拼接成tensor 得到最终输入gaussian_head的特征图 out:(BV, C_total', H, W)
        # C_total' = 64 + 3 + feature_upsampler_channels + 1（feature_upsampler_channels取决于不同的DiNOv2模型，分别是64，96，128。）
    
# 2.3.2 gaussian_head 用于预测初始Gaussian参数，输入是上一步融合的特征图，输出是高斯参数，包括尺度、旋转、球谐函数等。
        # 输入：融合特征图 out:(BV, C_total', H, W)
        # 输出：gaussians:(BV, C_g=37, H, W)，其中C_g是每个像素预测的高斯参数的数量，默认为 37。
        gaussians = self.gaussian_head(out)  # [BV, C_g=37, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v) # gaussians:(B, V, C_g=37, H, W)
# 2.4 把初始的高斯参数处理成每个 Gaussians 的参数
        # 把现有gaussians参数，深度图，密度图(可信度)展平成每个gaussian的性质。H*W 代表每张图上Gaussian的数量。
# 深度
        # (B, V, H*W, 1, 1)
        depths = rearrange(depth, "b v h w -> b v (h w) () ()") # depths:(B, V, H*W, 1, 1)，把深度图展平，这样每个像素就变成一个 ray sample

        # [B, V, H*W, 1, 1]
        densities = rearrange( # densities:(B, V, H*W, 1, 1)，深度可信度match_prob展平后得到的密度值，准备后续的高斯参数回归
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]  # 注意，在sh_degree=4的配置下才等于84。sh_degree=2时等于37。
        raw_gaussians = rearrange( # raw_gaussians:(B, V, H*W, C_g=37)，高斯参数展平后得到的原始高斯参数，准备后续的高斯参数回归
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1: # 中间深度监督（训练技巧）如果网络由多阶段的depth prediction，就可以就把这些 depth 一起监督。

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths
# 不透明度
        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)  # 取最后一个维度的第一个元素作为不透明度, sigmoid()保证0 ≤ opacity ≤ 1。
        # (B,V,H*W, C_g - 1=36)
        raw_gaussians = raw_gaussians[..., 1:] # 最后一个维度的剩下的部分是高斯参数，包括尺度、旋转、球谐函数等。
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device) # sample_image_grid()构建归一化的坐标。xy_ray:(H,W,2)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")  # xy_ray:(H,W,2) -> (H*W, 1, 2)，把坐标展平成每个像素一个 ray sample 的格式

# 高斯对应的位置(二维上的位置x,y)
        # gaussians:(B,V,H*W,1,C_g - 1=36)
        gaussians = rearrange( # Gaussian 参数重排
            raw_gaussians,  # (B,V,H*W, C_g-1=36)
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,  # 配置为1，表示每个像素预测一个高斯球
        )
        # offset_xy:(B,V,H*W,1,2)
        offset_xy = gaussians[..., :2].sigmoid() # 取最后一个维度的前两个元素作为x和y的偏移量，sigmoid()保证偏移量在0到1之间。
        # pixel_size = (1/W, 1/H)
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device) # 计算每个像素对应的实际大小，假设图像被归一化到[0,1]范围内，那么每个像素的大小就是1除以图像的宽高。
        # xy_ray:(B,V,H*W,1,2)，记录加上了偏移量后的真实位置。
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size # 计算真实 ray 位置
        # offset_xy - 0.5 : 把偏移量从[0,1]范围转换到[-0.5,0.5]范围，这样偏移就可以向四周扩展，而不是只在像素中心附近。
        # (offset_xy - 0.5) * pixel_size : 把偏移量缩放到实际的像素大小范围内，这样偏移就对应于图像上的实际距离。
        # xy_ray + 实际偏移量 : 最终得到每个像素对应的真实 ray 位置，考虑了偏移量和像素大小。

        sh_input_images = context["image"] # sh_input_images:(B,V,3,H,W)，
        # 如果配置了init_sh_input_img，就把输入图像也传给gaussian_adapter，用于初始化球谐函数的输入特征。

# 2.5 GaussianAdapter高斯适配器模块进行高斯参数回归（gaussian_adapter）：
        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1: # 如果有中间深度预测，就把这些中间深度对应的高斯参数也一起回归。
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )
    # gaussian_adapter.forward() 函数：
    # 输入：
    #        extrinsics:(B,V,1,1,1,4,4)，       相机外参，经过重排和扩展后，每个像素都有对应的外参。
    #        intrinsics:(B,V,1,1,1,3,3)，       相机内参，经过重排和扩展后，每个像素都有对应的内参。
    #        xy_ray:(B,V,H*W,1,2)，             每个像素的真实 ray 位置。（每个Gaussian对应的真实 像素位置。）
    #        depths:(B,V,H*W,1,1)，             每个像素的深度值。
    #        opacities:(B,V,H*W,1,1)，          每个像素的不透明度。
    #        gaussians:(B,V,H*W,1,1,C_g-1-2=34)   初始的高斯参数(-1：预测不透明度，-2：预测x,y的偏移量)。
    #        (h, w)：                           图像的高度和宽度。
    #        input_images:(B,V,3,H,W)           输入图像，用于初始化球谐函数的输入特征。（可选）
    # 
    # 输出：
    #       Gaussians(means, covariances, harmonics, opacities, scales, rotations)
    #           高斯位置：means: [B, V, H*W, 1, 1, 3]
    #           协方差：covariances: [B, V, H*W, 1, 1, 3, 3]
    #           轴向尺度：scales: [B, V, H*W, 1, 1, 3]
    #           旋转四元数：rotations: [B, V, H*W, 1, 1, 4]
    #           球谐系数：harmonics: [B, V, H*W, 1, 1, 3, 9]
    #           不透明度：opacities: [B, V, H*W, 1, 1]
        else: #（按配置走这里） 因为 len(depth_preds) = 1 
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],     # 前两个用于预测偏移量offset_xy了,剩下的部分是尺度、旋转、球谐函数等高斯参数，传给gaussian_adapter进行回归和转换。
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),         # 图像的高度和宽度。
                input_images=sh_input_images if self.cfg.init_sh_input_img else None, # 输入图像，用于初始化球谐函数的输入特征。
            )

        # Dump visualizations if needed.
        if visualization_dump is not None: # 把中间结果整理成可视化/调试用的数据，并存进 visualization_dump 字典里。
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # 把按“像素结构组织”的 Gaussian 展平为“渲染器需要的点列表形式”。
        #   r=H*W，即ray射线数。
        #   srf配置为 1，表示每条 ray 可能预测多个表面
        #   spp配置为 1，表示每个像素预测的高斯数量。
        # 最终得到的gaussians参数是一个列表，每个元素对应一个高斯参数，包括位置、尺度、旋转、球谐函数等，展平后的格式适合后续的渲染器使用。
        gaussians = Gaussians(
            rearrange(
                gaussians.means, # 位置：[B, V, H*W, 1, 1, 3]
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances, # 协方差：[B, V, H*W, 1, 1, 3, 3]
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics, # 球谐：[B, V, H*W, 1, 1, 3, 9]
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities, # 不透明：[B, V, H*W, 1, 1]
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
