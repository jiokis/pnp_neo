import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import BaseConfig


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 下采样分支
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down_proj = nn.Linear(input_dim, input_dim)

        # 上采样分支
        self.up_proj = nn.Linear(input_dim, input_dim)

        # 融合分支
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, src):
        """
        输入: src (B×V, N, input_dim)
        输出: fused_feat (B×V, N, output_dim)
        """
        BxV, N, _ = src.shape

        # 下采样分支
        src_permuted = src.permute(0, 2, 1)  # (B×V, C, N)
        downsampled = self.downsample(src_permuted)  # (B×V, C, N//2)
        downsampled = downsampled.permute(0, 2, 1)  # (B×V, N//2, C)
        down_feat = self.down_proj(downsampled)

        # 上采样分支
        upsampled = F.interpolate(down_feat.permute(0, 2, 1), size=N, mode='linear', align_corners=False)
        upsampled = upsampled.permute(0, 2, 1)  # (B×V, N, C)
        up_feat = self.up_proj(upsampled)

        # 特征融合
        concat_feat = torch.cat([src, up_feat], dim=-1)  # (B×V, N, 2C)
        fused_feat = self.fusion_mlp(concat_feat)

        return fused_feat


if __name__ == "__main__":
    # 测试多尺度融合
    fusion = MultiScaleFusion(input_dim=BaseConfig.LATENT_DIM).to(BaseConfig.DEVICE)
    src = torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.LATENT_DIM), device=BaseConfig.DEVICE)
    fused = fusion(src)
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {fused.shape}")