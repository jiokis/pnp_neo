import torch
import torch.nn as nn
from config.base_config import BaseConfig

class AnchorHead(nn.Module):
    """锚框匹配头（含置信度分支）"""
    def __init__(self, input_dim=128, num_anchors=4):
        super().__init__()
        self.num_anchors = num_anchors
        # 锚框匹配得分分支
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_anchors)
        )
        # 置信度分支
        self.conf_mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, latent_vecs):
        """
        输入: latent_vecs (B×V, N, 128)
        输出: match_scores (B×V, N, 4) - 锚框匹配得分
              conf_scores (B×V, N, 1) - 匹配置信度
        """
        # 锚框匹配得分
        match_scores = self.score_mlp(latent_vecs)
        # 置信度得分
        conf_scores = self.conf_mlp(latent_vecs)
        return match_scores, conf_scores

if __name__ == "__main__":
    # 测试锚框头
    head = AnchorHead(input_dim=BaseConfig.LATENT_DIM,
                     num_anchors=BaseConfig.ANCHOR_NUM).to(BaseConfig.DEVICE)
    latent_vecs = torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.LATENT_DIM), device=BaseConfig.DEVICE)
    match_scores, conf_scores = head(latent_vecs)
    print(f"输入形状: {latent_vecs.shape}")
    print(f"匹配得分形状: {match_scores.shape}")
    print(f"置信度形状: {conf_scores.shape}")