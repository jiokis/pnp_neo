import torch
import torch.nn as nn
from config.base_config import BaseConfig

class GeometricBranch(nn.Module):
    """几何特征分支"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super().__init__()
        self.geom_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ray_dir):
        """
        输入: ray_dir (B×V, N, 3)
        输出: geom_feat (B×V, N, 32)
        """
        # 射线方向编码
        geom_feat = self.geom_mlp(ray_dir)
        return geom_feat

if __name__ == "__main__":
    # 测试几何分支
    branch = GeometricBranch().to(BaseConfig.DEVICE)
    ray_dir = torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    geom_feat = branch(ray_dir)
    print(f"输入形状: {ray_dir.shape}")
    print(f"输出形状: {geom_feat.shape}")