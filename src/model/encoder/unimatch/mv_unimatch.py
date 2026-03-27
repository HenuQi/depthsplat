import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales     # 选择使用的特征图数量，默认为1，即只使用最低分辨率的特征图（1/8）。
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type   # 选择使用的 ViT 模型类型，默认为 "vits"，表示使用 ViT-Small 模型作为单目特征提取的 backbone。
        # 其他选项包括 "vitb"（ViT-Base）和 "vitl"（ViT-Large）。

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,   # 最低分辨率默认为8，说明输出的特征图分辨率最低为输入图像的1/8。
            return_all_scales=True,                   # 设置为True，说明需要返回所有分辨率的特征图列表，包含 1/2、1/4 和 1/8 分辨率的特征图。
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load( # 加载预训练的 DINOv2 模型，注意这里是在微调。
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused， 删除了用不到的 mask_token 以节省内存。

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )
        # 如果模型配置为多尺度（num_scales > 1），
        # 则使用 ViTFeaturePyramid 模块对提取出的 ViT 单目特征进行降采样/处理，
        # 生成不同分辨率的特征列表，以便与 CNN 的多尺度特征对齐。

        # UNet regressor
        self.regressor = nn.ModuleList()          # regressor 模块列表，包含每个尺度的 UNet 模块，用于处理融合后的特征图并预测深度。
        self.regressor_residual = nn.ModuleList() # regressor_residual 模块列表，包含每个尺度的残差连接模块，用于将输入特征图直接映射到 UNet 的输出通道数，以实现残差学习。
        self.depth_head = nn.ModuleList()         # depth_head 模块列表，包含每个尺度的深度预测头，用于从 UNet 的输出特征图中预测当前尺度的深度图。

        for i in range(self.num_scales):  # num_scales=1时，只循环一次，i=0
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)   # 128

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [ # Conv2d + GroupNorm + GELU：初步特征融合
                nn.Conv2d(in_channels, channels, 3, 1, 1), # 输入融合特征D+C+C+Cmono=768，输出128
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            # UNetModel ，UNet网络处理多尺度上下文信息
            modules.append(
                UNetModel(  # 输入128，输出128
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            # Conv2d 输出UNet网络处理后，深度预测前的特征图
            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))
            # 输入：
            # 输出：

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn
    # normalize_images()：ImageNet 归一化函数
    # 输入：images: [B, V, 3, H, W]
    # 输出：归一化后的 images:[B, V, 3, H, W]。此时每个通道的像素值已经根据 ImageNet 的均值和标准差进行了归一化处理。
    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    # extract_feature()：CNN 特征提取函数
    # 输入：images: (B, V, 3, H, W)
    # 输出: features list:[                     
    #                          1/8 features,   # (B*V,C=128, H/8, W/8)
    #                          1/4 features,   # (B*V,C=128, H/4, W/4)
    #                          1/2 features    # (B*V,C=128, H/2, W/2)               
    #                      ] 
    # 最终的输出特征图列表，分辨率从低到高 
    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w") # 多视图图像images:(B,V,C,H,W) -> CNN可以处理的concat:(B*V,C,H,W)
        # list of [BV, C, H, W], resolution from high to low

        # backbone()：CNNEncoder模块进行特征提取
        # 输入：合并B和V后的图像 concat:(B*V,3,H,W)
        # 输出：CNN提取的特征图列表 features list:[ 1/2 features, 1/4 features, 1/8 features ]
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1] # 反转特征图列表，变成从 低分辨率 到 高分辨率 features list:[ 1/8 features, 1/4 features, 1/2 features ]

        return features

# 2.定义前向传播函数forward()
    # 输入：(context) image:(B,V,3,H,W), 
    #       intrinsics:(B,V,3,3), 
    #       extrinsics:(B,V,4,4), 
    #       near:(B,V), 
    #       far:(B,V)
    # 输出：results_dict：{
    #           "features_cnn_all_scales": features_list_cnn_all_scales     # CNN提取的全部特征图列表 。                            每个元素:(B*V, C=128, H', W')，其中 H' 和 W' 是对应分辨率的大小，如 H/8,W/8 或 H/4,W/4 等。
    #           "features_cnn": features_list_cnn                           # CNN提取的只包含1/8分辨率特征图 的列表。                每个元素:(B*V, C=128, H/8, W/8)。
    #           "features_mv": features_list_mv                             # CNN+Transformer 提取融合的2D特征图列表。              每个元素:(B*V, C=128, H/8, W/8)。 
    #           "features_mono_intermediate": mono_intermediate_features    # DINOv2 的中间层特征图列表，包含了指定block的特征图。   每个元素:(B*V, C'=384, H/8, W/8)，其中C'是对应block的特征维度，如 ViT-S:384，ViT-B:768，ViT-L:1024。
    #           "features_mono": features_list_mono                         # DINOv2 的最后一个block的特征图（分辨率为1/8）。        每个元素:(B*V, C'=384, H/8, W/8)，其中C'是对应block的特征维度，如 ViT-S:384，ViT-B:768，ViT-L:1024。
    #           "depth_preds": depth_preds                                  # 深度预测的 深度图depth列表（已经从逆深度转换为深度）。  每个元素:(B, V, H, W)。
    #           "match_probs": match_probs                                  # 匹配概率图列表。                                      每个元素:(BV, D, H/8, W/8)，
    #       }
    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):

        results_dict = {}       # 存储中间结果的字典，包含特征图、深度预测等信息，供后续分析和可视化使用。
        depth_preds = []
        match_probs = []

# 2.1 ImageNet 归一化
        # first normalize images
        images = self.normalize_images(images) # images:(B,V,3,H,W)->(B,V,3,H,W)。 
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=v)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

# 2.2 CNN特征提取
        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images) # images:(B,V,3,H,W), features list:[ 1/8 features, 1/4 features, 1/2 features ]。
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales] # features_list_cnn:[1/8 features:(B*V,C,H/8,W/8)]。num_scales 默认为1，即只使用最低分辨率的特征图（1/8）。
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales}) # features_cnn_all_scales：CNN提取的全部特征图列表
        results_dict.update({"features_cnn": features_list_cnn}) # features_cnn ：CNN提取的只包含一个1/8分辨率特征图 的列表。

# 2.3 给CNN提取的特征图添加位置编码
        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0] # attn_splits默认为2，表示把特征分块（split）再加位置编码，即Window-based Attention / 分块注意力

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position( # 给CNN提取的特征图添加位置编码
            features_list_cnn[0], attn_splits, self.feature_channels
        )
        # features_list_cnn[0]:(B*V,C,H/8,W/8) -> features_cnn_pos:(B*V,C,H/8,W/8)。shape不变，只是加上了位置编码。


        # list of [B, C, H, W]
        features_list = list( # 将加了位置编码的特征图按照视图维度分开，得到一个包含每个视图特征图的列表，供后续的多视图Transformer处理。
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        # features_cnn_pos:(B*V,C,H/8,W/8)
        # features_list:[
        #     (B, C, H/8, W/8),  # view 1
        #     (B, C, H/8, W/8),  # view 2
        #     ...]
        # 里面的每一个元素都是各个视图的添加了位置编码后的特征图。

# 2.4 多视图Transformer特征提取
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = rearrange( # 将多视图Transformer处理后的特征图列表 堆叠成一个张量features_mv(B*V,C,H/8,W/8)
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]
        # features_mv 即为 CNN+Transformer 提取融合的2D特征图（分辨率为1/8，包含了多视图信息）

        if self.num_scales > 1: 
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            features_list_mv = self.mv_pyramid(features_mv)
        else: # （默认）
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv}) # 特征图列表，里面只包含CNN+Transformer 提取融合的2D特征图
        # features_list_mv : [ (B*V,C,H/8,W/8) ]。如果 num_scales > 1，则还会包含更高分辨率的特征图（如1/4、1/2等）。

# 2.5 Dinov2 单目特征提取
        # mono feature
        ori_h, ori_w = images.shape[-2:]
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14 
        concat = rearrange(images, "b v c h w -> (b v) c h w") # 合并后的图像concat:(B*V,3,H,W)
        concat = F.interpolate( # 对视图进行尺寸缩放（双线性插值） 
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        ) # concat:(B*V,3,H,W) -> (B*V,3,H/14,W/14)。
        # DINOv2 的 Patch Size 是 14，因此将输入图像的尺寸调整为 14 的倍数，以适配 DINOv2 的输入要求。

        # get intermediate features   不只是取 DINOv2 最后一层的输出，而是提取特定中间层的特征
        intermediate_layer_idx = { # 取哪几层的特征
            "vits": [2, 5, 8, 11],    # 取第3，6，9，12层的特征（从0开始）
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        # DINOv2 的预训练模型带有的方法get_intermediate_layers():返回中间层特征
        # 从ViT Transformer的指定block中提取patch token特征
        # 输入：合并后的图像 concat:(B*V,3,H,W)
        # 输出：list of [B*V, N, C=384]。其中N是Token的数量(N=H/14 × W/14)，C是每个Token的特征维度 (ViT-S:384, ViT-B:768, ViT-L:1024)。
        # mono_intermediate_features = [
        #                                   block3 feature,     # (B*V, N, C)
        #                                   block6 feature,     # (B*V, N, C)
        #                                   block9 feature,     # (B*V, N, C)
        #                                   block12 feature     # (B*V, N, C)  最后一个block的输出特征
        #                               ]
        # 注意：return_class_token=False，返回patch tokens。return_class_token=True，返回(cls_token, patch_tokens)
        mono_intermediate_features = list( # 取中间层的特征
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )

        # 将 mono_intermediate_features列表中的 1D Token 序列重塑为 2D 空间特征图
        for i in range(len(mono_intermediate_features)): 
            curr_features = ( # curr_features -> (B*V, C, H/14, W/14)
                mono_intermediate_features[i]               # mono_intermediate_features[i]:(B*V, N, C) ->
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)   # (B*V, H/14, W/14, C) ->
                .permute(0, 3, 1, 2)                                            # (B*V, C, H/14, W/14)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate( # curr_features:(B*V, C, H/14, W/14) -> (B*V, C, H/8, W/8)
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features
        # 循环结束后得到的：
        # mono_intermediate_features = [
        #                                   block3 feature map,     # (B*V, C, H/8, W/8)
        #                                   block6 feature map,     # (B*V, C, H/8, W/8)
        #                                   block9 feature map,     # (B*V, C, H/8, W/8)
        #                                   block12 feature map     # (B*V, C, H/8, W/8)  最后一个block的输出特征图
        #                               ]
        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature
        mono_features = mono_intermediate_features[-1] # mono_features:(B*V,C,H/8,W/8)。取最后一个block的输出特征图作为单目特征

        if self.lowest_feature_resolution == 4: # 特殊情况：如果要求最低分辨率为1/4
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1: # 如果模型配置为多尺度（num_scales > 1），则使用 ViTFeaturePyramid 模块对提取出的 ViT 单目特征进行降采样/处理，生成不同分辨率的特征列表，以便与 CNN 的多尺度特征对齐。
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            features_list_mono = self.mono_pyramid(mono_features)
        else: #（默认） 单目特征列表 features_list_mono 只包含最后一个block的单目特征图mono_features。
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})
        # features_list_mono: [ (B*V,C,H/8,W/8) ]

# 2.6 多视图匹配构建代价体 (下面的所有 H 和 W 都指代的是对应分辨率的大小，如 H/8,W/8) 
        depth = None

        for scale_idx in range(self.num_scales): # 从粗到细（Coarse-to-Fine）多尺度多视图立体匹配（MVS）
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            ) # scale_idx=0 时，downsample_factor=8 * 1=8 不变，即上采样8倍
# 2.6.1 预处理工作一，调整当前尺度的相机内参矩阵。
            # scale intrinsics  # 调整当前尺度的相机内参矩阵。
            # 因为网络是在降采样后的特征图（如原图的 1/8分辨率）上进行操作的，图像的宽和高缩小了，
            # 所以相机的焦距 fx,fy 和 光心坐标 cx,cy 必须等比例缩小(即内参)，以保持正确的投影关系。
            intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor
# 2.6.1 预处理工作二，把特征图，内参，外参按照视图维度分开（BV -> B,V）
            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]
            # 取出当前尺度的特征图 features_mv:(B*V,C=128,H/8,W/8)

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
                )
            )
            # list of [B, C, H, W]， 长度为V
            # features_mv_curr ：list of [B, C, H, W]， 长度为V

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3] 
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]
            # intrinsics_curr : list of [B, 3, 3] ,长度为 V
            # extrinsics_curr : list of [B, 4, 4] ,长度为 V
# 2.6.1 预处理工作三，选定一个视图作为参考帧 (Ref), 并处理好用于计算的特征图、相机内参和外参
            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters( # 选定一个视图作为参考帧 (Ref)，其余 V-1 个视图作为目标帧 (Tgt)
                features_mv_curr,               # list of (B,C=128,H/8,W/8)，长度为V
                intrinsics_curr,                # list of (B,3,3)，长度为V
                extrinsics_curr,                # list of (B,4,4)，长度为V
                nn_matrix=nn_matrix,            
            )

            b_new, _, c, h, w = tgt_features.size() # tgt_features:(B*V,V-1,C=128,H/8,W/8)

            # relative pose
            # extrinsics: c2w
# 2.6.1 预处理工作四，计算从参考相机坐标系到目标相机坐标系的变换矩阵 T(Cref -> Ctgt)，供后续特征 Warping 使用
            pose_curr = torch.matmul( # 计算从参考相机坐标系到目标相机坐标系的变换矩阵 T(Cref -> Ctgt)
                tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)  # tgt_c2w @ ref_w2c = T(Cref -> Ctgt)
            )  # [BV, V-1, 4, 4]

            if scale_idx > 0: # 在多尺度深度预测过程中，把上一层预测的深度图上采样到当前尺度，并且阻断梯度传播。
                # 2x upsample depth
                assert depth is not None # 确保有上一次预测的深度结果
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

# 2.6.2 生成候选深度 (Depth Candidates)    
            num_depth_candidates = self.num_depth_candidates // (4**scale_idx) # 候选深度个数 D 默认为128，（分辨率1/8时）
            # 每增加一个尺度，候选深度个数减少4倍（即间隔加大），以适应更高分辨率特征图的细化需求。

            # generate depth candidates
            if scale_idx == 0:  # （默认）如果是单尺度深度预测，即1/8分辨率
                # min_depth, max_depth: [BV]
                # 计算深度采样间隔 = 深度范围/（候选深度个数-1）
                depth_interval = (max_depth - min_depth) / (  # max_depth：[BV]，min_depth:[BV]，depth_interval:[BV]
                    self.num_depth_candidates - 1
                )  # [BV]

                # 生成 [0,1] 的线性空间。定义D个深度把空间分割成D-1段，每段对应一个深度采样间隔
                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates) # .linspace()生成一个从 0 到 1 的 等间隔序列, 如:torch.linspace(0, 1, 5) 生成 [0., 0.25, 0.5, 0.75, 1.]
                    .type_as(features_list_cnn[0])  # 保证这个张量的数据类型和 CNN 特征图一致
                    .view(1, num_depth_candidates, 1, 1) # 把 一维张量(D,)  reshape 成 (1, D, 1, 1)
                )  # [1, D, 1, 1]

                # 生成候选深度 depth candidates:(BV,D,1,1)，表示每张视图都有 D 个候选深度值
                depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                    max_depth - min_depth
                ).view(
                    -1, 1, 1, 1
                )  # [BV, D, 1, 1]
                #  linear_space：               shape为(1,D,1,1)，表示在[0,1]的深度范围内等间隔的 D 个采样点的位置（从0到1的比例）。      
                # (max_depth - min_depth)：     shape为(BV,1,1,1),表示实际上的深度范围
                # 二者相乘的结果：               shape为(BV,D,1,1)，表示在实际的深度范围内等间隔的 D 个候选深度值。   
                # 最后加上 min_depth:(BV,1,1,1), 得到最终的候选深度值 depth_candidates:(BV,D,1,1)   
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  # [BV]
                # [BV, 1, 1, 1]
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                # [BV, 1, H, W]
                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  # [BV, D, H, W]

            if scale_idx == 0: # （默认）如果是单尺度深度预测，即1/8分辨率
                # [BV*(V-1), D, H, W]
                depth_candidates_curr = ( # 将 depth candidates 扩展到 每个tgt目标视图（特征）
                    depth_candidates.unsqueeze(1)   # depth_candidates: (BV,D,1,1) -> (BV,1,D,1,1)
                    .repeat(1, tgt_features.size(1), 1, h, w) # -> (BV,V-1,D,H,W)，即每个目标视图都使用相同的候选深度值
                    .view(-1, num_depth_candidates, h, w)     # -> (BV*(V-1),D,H,W)  把前两维(BV,V-1)合并，方便后续的特征 Warping 操作
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

# 2.6.3 特征 Warping
            # 输入内参扩展到 每个tgt目标视图（特征）
            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BV, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BV, V-1, 3, 3]

            # Wraping操作： Fj -> Fi, 得到每个目标视图（特征）在参考视图坐标系下的投影特征图 
            # 即论文中的 Fj->i = warped_tgt_features : (BV*(V-1), C=128, D=128, H/8, W/8)
            warped_tgt_features = warp_with_pose_depth_candidates( # 对目标特征进行投影
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,  # convert inverse depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BV*(V-1), C, D, H, W]

            # ref: [BV, C, H, W]
            # warped: [BV*(V-1), C, D, H, W] -> [BV, V-1, C, D, H, W]
            warped_tgt_features = rearrange( # 恢复 batch 和视图维度
                warped_tgt_features,
                "(b v) ... -> b v ...",
                b=b_new,
                v=tgt_features.size(1),
            ) 
            # 此时 warped_tgt_features:(BV, V-1, C=128, D=128, H/8, W/8)，表示每个目标视图在参考视图坐标系下的投影特征图，包含了深度维度 D。

# 2.6.4 构建代价体积 cost_volume
            # [BV, V-1, D, H, W] -> [BV, D, H, W]
            # average cross other views
            cost_volume = (
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5) # c=128, 除以特征维度的平方根，用于缩放
            ).mean(1)
            # warped_tgt_features:(BV, V-1, C=128, D=128, H/8, W/8)
            # ref_features:(BV,C=128,H/8,W/8) -> (BV, 1, C=128, 1, H/8, W/8)
            # 二者相乘再在C维度上相加，即逐像素的向量 做点积，结果可以衡量两张特征图之间的相似度（即相似度图）
                # 相乘之后 -> (BV, V-1, C, D, H/8, W/8) ；相加之后 -> (BV, V-1, D, H/8, W/8) 
                # 点积：向量内各个元素相乘，最后相加。值越大，表示两个向量之间越相似。
                # 两张特征图做点积，得到一张相似度图。相似度图中某一像素的值越大，说明两张图在该位置越相似。
            # / (c**0.5)： 把相似度图 除以特征维度通道数的平方根 进行缩放，让匹配分数在不同通道数量下保持可比性。
            # .mean(1)： 在视图维度上取平均，得到最终的代价体积。  (BV, V-1, D, H/8, W/8) -> (BV, D, H/8, W/8)
            # 最终得到视图 i 的代价体 cost_volume:(BV, D=128, H/8, W/8)，表示每个像素在不同深度候选下的匹配成本（相似度）。深度候选越接近真实深度，匹配成本越低（相似度越高）。
# 2.7 深度预测
# 2.7.1 拼接所有特征 并 进行特征融合
            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            # 拼接特征
            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
                # cost_volume:(BV, D=128, H/8, W/8)
                # features_cnn:(BV, C=128, H/8, W/8)
                # features_mv:(BV, C=128, H/8, W/8)
                # features_mono:(BV, C=384, H/8, W/8)
            ) # concat:(BV, C_total = D+C+C+Cmono, H/8, W/8)

            # 融合特征
            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)
            # .regressor()：对拼接的特征进行卷积融合，输出：特征图 out:(BV, D=128, H/8, W/8)
            # .regressor_residual()：对拼接的特征进行卷积处理，输出：残差特征图 out_residual:(BV, D=128, H/8, W/8)，作为残差连接的部分。
            # residual连接：把 regressor 的输出和 regressor_residual 的输出逐元素相加，得到最终的融合特征图 out。这种残差连接有助于模型更好地学习特征融合。
            # 得到用于深度预测的特征图 out：(B*V, C=128, H/8, W/8)

# 2.7.2 从融合特征图 out 中预测深度
            # 计算候选深度概率 match_prob:(BV, D, H/8, W/8)
            # depth pred
            match_prob = F.softmax(
                self.depth_head[scale_idx](out), dim=1
            )  # [BV, D, H, W]
            # depth_head 是一个卷积层（Conv2d）,
            # 输出 [BV, D, H/8, W/8]，D 是当前尺度的深度候选数量,
            # 每个像素 (h,w) 上的值对应每个深度候选的“匹配概率”，表示此像素是在这个候选深度下的概率
            # 注意：某个像素上，深度概率之和为 1 （即：某一个像素对应的向量，内部元素之和为1）
            match_probs.append(match_prob)

            # (默认）如果是单尺度深度预测直接计算深度图（候选深度 和 其对应的概率 加权求和）
            # 计算深度图 depth:(BV, 1, H/8, W/8)
            if scale_idx == 0:
                # [BV, D, H, W]
                depth_candidates = depth_candidates.repeat(1, 1, h, w) # depth_candidates:[BV, D, 1, 1]->[BV, D, H/8, W/8]。把候选深度值扩展到每个像素位置，方便后续的加权求和。
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]

            # （默认不走）多尺度训练时上采样到高分辨率
            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

# 2.7.3 最终输出优化且上采样了的深度图（此时还是逆深度）
            # 如果是单尺度深度预测，直接使用 learned upsampler 进行上采样和细化，
            # 如果是多尺度深度预测，则在循环到最后一个尺度上时候上采样，得到最终的深度图。
            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(  # upsampler 是一个 DPTHead 学习型上采样模块，输出 residual_depth → 表示 细化残差，用于修正深度图
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv if self.num_scales == 1 else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate( # 此处是双线性上采样到原始分辨率，作为粗略的深度图
                    depth,
                    scale_factor=self.upsample_factor, # upsample_factor默认为8，即从1/8分辨率上采样到原始分辨率
                    mode="bilinear",
                    align_corners=True,
                )
                # 将双线性上采样的深度图 depth_bilinear 与细化残差residual_depth相加，并在指定范围内进行裁剪
                # 得到最终的深度图 depth:(B*V, 1, H, W)
                depth = (depth_bilinear + residual_depth).clamp(
                    min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
                )

                depth_preds.append(depth)
# 把预测的深度图从逆深度转换为深度
        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BV, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )  # [B, V, H, W]
        # 得到的 depth_preds 是一个列表，里面包含了每个尺度预测的深度图（已经从逆深度转换为深度），
        # 最终的 depth 的 shape 是 (B, V, H, W)，表示每个视图的深度图。

        results_dict.update({"depth_preds": depth_preds})
        # depth_preds:[ (B, V, H, W) ]，    是不同尺度的深度图的列表，已经从逆深度转换为深度。
        results_dict.update({"match_probs": match_probs})
        # match_probs:[ (BV, D, H/8, W/8) ]， 是不同尺度的匹配概率图的列表

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
