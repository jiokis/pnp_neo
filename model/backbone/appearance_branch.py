import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config.base_config import BaseConfig
from model.lora.lora_adaptor import LoRACameraAdaptor

class MobileNetV3AppearanceBranch(nn.Module):
    """MobileNetV3-Tiny外观特征分支（集成特征金字塔）"""

    def __init__(self, output_dim=96):
        super().__init__()
        # 加载预训练MobileNetV3-Tiny
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # 冻结前3层
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < 3:
                param.requires_grad = False
            else:
                break

        # 新增：提取MobileNetV3的多尺度特征层（根据实际结构调整）
        self.features = self.backbone.features  # 保留原始特征提取序列
        # 注册钩子获取中间层特征（假设第3/6/9层是不同尺度的关键特征）
        self.multi_scale_layers = [3, 6, 9]  # 需根据实际打印的层结构微调
        self.multi_scale_feats = []

        # 新增：特征金字塔（最大池化版）
        self.pyramid = LightweightFeaturePyramid(
            in_channels=160,
            out_channels=128  # 输出通道可根据需要调整
        )
        self.lora = LoRACameraAdaptor(in_dim=output_dim, rank=8)

        # 替换最后一层，输出指定维度
        self.feature_dim = 128  # 金字塔融合后的通道数
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # 新增：注册钩子获取中间特征
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子获取多尺度中间特征"""

        def get_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx in self.multi_scale_layers:
                    self.multi_scale_feats.append(output)

            return hook

        # 给特征层注册钩子
        for idx, layer in enumerate(self.features):
            layer.register_forward_hook(get_hook(idx))

    def forward(self, image, num_rays):
        """
        输入: image (B×V, C, H, W)
              num_rays: 射线数
        输出: appear_feat (B×V, N, 96)
        """
        BxV = image.shape[0]
        self.multi_scale_feats = []  # 清空缓存

        # 1. 提取多尺度特征（通过钩子获取）
        _ = self.features(image)  # 前向传播触发钩子
        # 确保获取到3个尺度的特征（若不足则用最大池化补充）
        while len(self.multi_scale_feats) < 3:
            self.multi_scale_feats.append(F.max_pool2d(self.multi_scale_feats[-1], 2, 2))

        # 2. 特征金字塔融合
        # 统一特征尺寸（以最小尺寸为准，或插值到同一尺寸）
        target_h, target_w = self.multi_scale_feats[-1].shape[2:]
        aligned_feats = []
        for feat in self.multi_scale_feats:
            aligned = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            aligned_feats.append(aligned)
        # 拼接并融合
        fused_feat = self.pyramid(torch.cat(aligned_feats, dim=1))  # 输入通道数需匹配

        # 3. 全局特征提取（替换原来的自适应池化）
        x = F.adaptive_avg_pool2d(fused_feat, (1, 1))
        x = torch.flatten(x, 1)  # (B×V, feature_dim)

        # 4. 特征投影
        global_feat = self.head(x)  # (B×V, 96)
        # 新增：LORA微调（就在这里！）
        global_feat = self.lora(global_feat)


        # 5. 扩展到射线维度
        appear_feat = global_feat.unsqueeze(1).repeat_interleave(num_rays, dim=1)
        return appear_feat


# 最大池化特征金字塔（复用之前的实现）
class LightweightFeaturePyramid(nn.Module):
    def __init__(self, in_channels=160, out_channels=128):  # 输入改为160，输出可自定义
        super().__init__()
        self.down1 = nn.MaxPool2d(2, stride=2)
        self.down2 = nn.MaxPool2d(4, stride=4)
        self.fuse = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # 输入通道改为160

    def forward(self, x):
        return self.fuse(x)


# LORA模块实现（复用之前的代码）


if __name__ == "__main__":
    # 测试外观分支
    branch = MobileNetV3AppearanceBranch().to(BaseConfig.DEVICE)
    image = torch.randn((8, 3, BaseConfig.IMG_HEIGHT, BaseConfig.IMG_WIDTH), device=BaseConfig.DEVICE)
    appear_feat = branch(image, num_rays=BaseConfig.NUM_RAYS)
    print(f"输入形状: {image.shape}")
    print(f"输出形状: {appear_feat.shape}")