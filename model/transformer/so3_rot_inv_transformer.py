import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import BaseConfig


class SO3RotInvTransformer(nn.Module):
    """距离感知旋转不变Transformer"""

    def __init__(self, d_model=128, nhead=1, local_radius=0.6):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.local_radius = local_radius

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 相对方向编码
        self.rel_dir_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, nhead)
        )

        # 特征投影
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        # 特征区分投影
        self.distinct_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, src, ray_3d, ray_dir):
        """
        输入: src (B×V, N, d_model)
              ray_3d (B×V, N, 3) - 3D点坐标
              ray_dir (B×V, N, 3) - 射线方向
        输出: out (B×V, N, d_model)
        """
        BxV, N, _ = src.shape
        device = src.device

        # 计算局部注意力掩码
        coord_dist = torch.cdist(ray_3d, ray_3d)  # (B×V, N, N)
        local_mask = coord_dist < self.local_radius  # (B×V, N, N)

        # 计算相对方向编码
        rel_dir = ray_dir.unsqueeze(2) - ray_dir.unsqueeze(1)  # (B×V, N, N, 3)
        rel_dir = F.normalize(rel_dir, dim=-1)
        rel_enc = self.rel_dir_mlp(rel_dir)  # (B×V, N, N, nhead)

        # 生成查询、键、值
        q = self.proj_q(src)  # (B×V, N, d_model)
        k = self.proj_k(src)  # (B×V, N, d_model)
        v = self.proj_v(src)  # (B×V, N, d_model)

        # 计算注意力权重
        attn_output_weights = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.d_model, device=device))
        # 融入相对方向编码
        attn_output_weights = attn_output_weights.view(BxV, N, N, self.nhead) + rel_enc
        attn_output_weights = attn_output_weights.view(BxV, N, N)

        # 距离衰减（远距点注意力权重降低）
        dist_decay = 1 - (coord_dist / (coord_dist.max(dim=-1, keepdim=True)[0] + 1e-8))
        attn_output_weights = attn_output_weights * dist_decay

        # 应用局部掩码
        min_val = torch.finfo(attn_output_weights.dtype).min
        attn_output_weights = attn_output_weights.masked_fill(~local_mask, min_val)

        # 多头注意力
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output, _ = self.multihead_attn(q, k, v, attn_mask=None)

        # 残差连接+特征区分
        out = src + attn_output
        out = self.distinct_proj(out)

        return out


if __name__ == "__main__":
    # 测试Transformer
    transformer = SO3RotInvTransformer(d_model=BaseConfig.LATENT_DIM,
                                       nhead=BaseConfig.TRANSFORMER_HEADS,
                                       local_radius=BaseConfig.LOCAL_RADIUS).to(BaseConfig.DEVICE)
    src = torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.LATENT_DIM), device=BaseConfig.DEVICE)
    ray_3d = torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    ray_dir = torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    out = transformer(src, ray_3d, ray_dir)
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {out.shape}")