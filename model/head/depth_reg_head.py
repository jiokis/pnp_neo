import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import BaseConfig

class DepthRegressionHead(nn.Module):
    """深度回归头（已适配128维特征）"""
    def __init__(self, input_dim=128, num_anchors=16):
        super().__init__()
        self.input_dim = input_dim  # 固定为128（与latent_vecs特征维度一致）
        self.num_anchors = num_anchors
        self.depth_min = BaseConfig.MIN_DEPTH
        self.depth_max = BaseConfig.MAX_DEPTH

        # 生成锚点中心
        self.anchor_centers = self._generate_anchor_centers()

        # 回归器（输入维度=128，彻底解决矩阵乘法错误）
        self.regressor = nn.Sequential(
            nn.Linear(self.input_dim, 128),  # 128×128，与输入特征匹配
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def _generate_anchor_centers(self):
        return torch.linspace(self.depth_min, self.depth_max, self.num_anchors)

    def forward(self, latent_vecs, anchor_idx):
        # 设备对齐
        device = latent_vecs.device
        anchor_centers = self.anchor_centers.to(device)

        # 锚点处理（形状：B×V=4, N=512 → 展开为2048×512）
        B_V, N = anchor_idx.shape
        anchor_centers_expand = anchor_centers.unsqueeze(0).unsqueeze(0).expand(B_V, N, -1)
        anchor_idx_expand = anchor_idx.unsqueeze(-1).long()
        selected_anchor = torch.gather(anchor_centers_expand, dim=-1, index=anchor_idx_expand).squeeze(-1)

        # 残差预测（关键：latent_vecs先展平为2048×128，适配线性层）
        latent_vecs_flat = latent_vecs.reshape(-1, self.input_dim)  # 4×512×128 → 2048×128
        residual_depth_flat = self.regressor(latent_vecs_flat).squeeze(-1)  # 2048×1
        residual_depth = residual_depth_flat.reshape(B_V, N)  # 恢复为4×512

        # 最终深度预测
        pred_depth = torch.clamp(selected_anchor + residual_depth, self.depth_min, self.depth_max)
        return pred_depth, residual_depth