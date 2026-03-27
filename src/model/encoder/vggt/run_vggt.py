# 用aggregator提取特征features: list of (B, V, P, 2C)
# 从demo_colmap.py中学习

import torch
import torch.nn as nn
import torch.nn.functional as F


# from vggt.models.vggt import VGGT

# 图片处理成适配VGGT输入格式的函数(VGGT固定输入尺寸为518*518)
def preprocess_images_for_vggt(images, resolution=518):
    # images: [B, V, 3, H, W]  H=W=256

    # 确保输入图像具有正确的形状和通道数
    assert len(images.shape) == 5
    assert images.shape[2] == 3

    B, V, C, H, W = images.shape
    images = images.reshape(B * V, C, H, W) # images: [B, V, 3, H, W]不能直接插值，需要reshape成[BV, 3, H, W]才可以
    # [B, V, 3, H, W] -> [BV, 3, H, W]

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    # VGGT 固定输入尺寸为518*518。需要把图片用双线性插值，resize到518*518
    # images：[N, 3, 518, 518]

    images = images.reshape(B, V, C, resolution, resolution) # 把图片reshape回[B, V, 3, 518, 518]，以适应后续的VGGT输入格式
    # [BV, 3, H, W] -> [B, V, 3, 518, 518]

    return images


# run_Aggregator()函数作用:用aggregator提取特征features: list of (B, V, P, 2C), 类似demo_colmap.py中的run_VGGT()
def run_Aggregator(aggregator, images, resolution=518):
    # images: [B, V, 3, 518, 518]

# # ------------------------------------------------有了with torch.no_grad():，就不用这一部分了？？先注释掉
# # 调用冻结的model.aggregator() 提取特征，
#     # 再次冻结aggregator的参数，确保在微调训练过程中不更新aggregator的权重。
#     for param in aggregator.parameters():
#         param.requires_grad = False
#     aggregator.eval()
# #  -------------------------------------------------
    aggregator.eval()
    
    # 调用model.aggregator() 提取特征tokens
    with torch.no_grad():
        # with torch.cuda.amp.autocast(dtype=dtype):  # 原版： with torch.cuda.amp.autocast(dtype=dtype)    不过这里没有定义dtype是什么，所以直接用默认的autocast()，让PyTorch自动选择合适的精度（通常是float16），以提高推理效率。
        aggregated_tokens_list, ps_idx = aggregator(images)
    # 输出：
    #   aggregated_tokens_list：
    #       multi-scales token的列表,
    #       即：长度为24的 list of (B, V, P, 2C)，每个元素代表一个 AA block的输出（类似ViT feature）
    #   ps_idx：
    #       patch token 在序列中的起始下标（int），用于切分掉 camera/register token
    return aggregated_tokens_list, ps_idx



# run_DepthHead()函数作用:用depth_head进行深度图预测,得到 depth_map, depth_conf 
def run_DepthHead(depth_head, aggregated_tokens_list, images, patch_start_idx):
    # images: [B, V, 3, 518, 518]
    
    ######## 新增了feature_map
    depth_map, depth_conf, feature_map= depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
    # 输出：
    #   depth_map: [B, V, num_depth_candidates]，每个像素的深度候选值
    #   depth_conf: [B, V, num_depth_candidates]，每个深度候选值的置信度
    #   feature_map: [B, V, C, H, W]，从depth_head提取的特征图

    return depth_map, depth_conf, feature_map   ######### 新增了feature_map 


# 没用了
# 处理token ,从(B, V, P, 2C) -> (B, V, C, H, W)
# # 可以利用 depth_head ，设置只输出特征图，不经过最后的卷积层和激活头，得到 (B, V, C, H, W) 的特征图，用于高斯参数回归。
# def extract_features_from_tokens(depth_head, aggregated_tokens_list, images, patch_start_idx):
#     # images: [B, V, 3, 518, 518]

#     features = depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
#     # 输出：
#     #   features: (B, V, C, H, W)，从depth_head提取的特征图，用于高斯参数回归

#     return features


# 处理depth_map和depth_conf
def preprocess_depth_for_depthsplat(depth_map, depth_conf, resolution=256):

    B, V, H, W, C = depth_map.shape
    depth_map = depth_map.reshape(B * V, C, H, W) # depth_map: [B, V, H, W, 1]不能直接插值，需要reshape成[BV, C, H, W]才可以
    depth_conf = depth_conf.reshape(B * V, C, H, W) # depth_conf: [B, V, H, W, 1]不能直接插值，需要reshape成[BV, C, H, W]才可以

    # 插值到256*256，适配DepthSplat的输入要求
    depth_map_resized = F.interpolate(depth_map, size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
    depth_conf_resized = F.interpolate(depth_conf, size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True) 

    depth_map_resized = depth_map_resized.reshape(B, V, resolution, resolution, C) # 把图片depth_map_resized回[B, V, 256, 256, C ]
    depth_conf_resized = depth_conf_resized.reshape(B, V, resolution, resolution, C) # 把图片depth_conf_resized回[B, V, 256, 256, C]

    return depth_map_resized, depth_conf_resized

# 处理feature_map
def preprocess_features_for_depthsplat(feature_downsampler, feature_map):

    B, V, C, H, W = feature_map.shape
    feature_map = feature_map.reshape(B * V, C, H, W) # feature_map: [B, V, C, H, W]不能用BlurDownsample下采样，需要reshape成[BV, C, H, W]才可以

    features_downsampled = feature_downsampler(feature_map)  # feature_map: (BV, C, H, W) -> features_downsampled: (BV, C, 256, 256)   

    # 把图片features_downsampled回[B, V, C, 256, 256]
    features_downsampled = features_downsampled.reshape(B, V, features_downsampled.shape[1], features_downsampled.shape[2], features_downsampled.shape[3])    

    return features_downsampled

# class BlurDownsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         self.conv = nn.Conv2d(channels, channels, 3, padding=1)

#         # 简单 blur kernel
#         kernel = torch.tensor([1., 2., 1.])
#         kernel = kernel[:, None] * kernel[None, :]
#         kernel = kernel / kernel.sum()
#         self.register_buffer("kernel", kernel[None, None, :, :])

#     def forward(self, x):
#         # ！！！加上这两行！！！
#         x = self.conv(x)          # 先让输入经过卷积层提取和细化特征
#         x = F.relu(x)             # 加上激活函数，增加网络的非线性表达能力（推荐）

#         B, C, H, W = x.shape

#         kernel = self.kernel.repeat(C, 1, 1, 1)

#         # blur
#         x = F.conv2d(x, kernel, padding=1, groups=C)

#         # downsample downsample (518 -> 259)
#         x = x[:, :, ::2, ::2]

#         # 对齐到256 (259 -> 256) 用 adaptive_avg_pool2d 是可以跑通的，只是边缘会有极轻微的不均匀。
#         # x = F.adaptive_avg_pool2d(x, (256, 256))
#         # # 也可以用插值下采样到256*256，但可能会有轻微的模糊。根据实际效果选择。
#         x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False) 

#         return x
    
class BlurDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

        # blur kernel
        kernel = torch.tensor([1., 2., 1.])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :])

    def forward(self, x):
        B, C, H, W = x.shape

        kernel = self.kernel.repeat(C, 1, 1, 1)

        # 1. anti-alias blur
        x = F.conv2d(x, kernel, padding=1, groups=C)

        # 2. downsample（更推荐 avg_pool）
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 518 → 259

        # 3. learnable refinement
        x = self.act(self.conv(x))

        # 4. 精确对齐（不用插值！）
        x = F.adaptive_avg_pool2d(x, (256, 256))

        return x