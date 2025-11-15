import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import BaseConfig


class DepthCalibration(nn.Module):
    """深度校准模块（后处理）"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # 相邻点平滑卷积
        self.smooth_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        # 全局缩放因子
        self.global_scale = nn.Parameter(torch.ones(1))

    def forward(self, pred_depth, ray_3d):
        """
        输入: pred_depth (B×V, N) - 原始预测深度
              ray_3d (B×V, N, 3) - 3D点坐标
        输出: calib_depth (B×V, N) - 校准后深度
        """
        BxV, N = pred_depth.shape
        device = pred_depth.device

        # 1. 全局缩放校准
        scaled_depth = pred_depth * self.global_scale

        # 2. 相邻点平滑
        depth_permuted = scaled_depth.unsqueeze(1)  # (B×V, 1, N)
        smoothed_depth = self.smooth_conv(depth_permuted).squeeze(1)  # (B×V, N)

        # 3. 基于3D距离的加权平滑
        coord_dist = torch.cdist(ray_3d, ray_3d)  # (B×V, N, N)
        # 生成距离权重（高斯核）
        sigma = 0.5
        dist_weight = torch.exp(-coord_dist ** 2 / (2 * sigma ** 2))  # (B×V, N, N)
        # 应用权重
        weighted_depth = torch.bmm(dist_weight, smoothed_depth.unsqueeze(-1)).squeeze(-1)

        # 4. 结果限制
        calib_depth = torch.clamp(weighted_depth, BaseConfig.MIN_DEPTH, BaseConfig.MAX_DEPTH)

        return calib_depth


if __name__ == "__main__":
    # 测试深度校准
    calib = DepthCalibration().to(BaseConfig.DEVICE)
    pred_depth = torch.randn((8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE) * 5 + 15
    ray_3d = torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    calib_depth = calib(pred_depth, ray_3d)
    print(f"输入深度形状: {pred_depth.shape}")
    print(f"校准深度形状: {calib_depth.shape}")
    print(f"校准前范围: [{pred_depth.min():.2f}, {pred_depth.max():.2f}]")
    print(f"校准后范围: [{calib_depth.min():.2f}, {calib_depth.max():.2f}]")