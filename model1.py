import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalGeomMLP(nn.Module):
    """精简3D几何分支：输入射线方向，输出32维特征"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )

    def forward(self, ray_dirs):
        return self.mlp(ray_dirs)


class MinimalAppearBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # (3,480,640)→(16,240,320)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # →(32,120,160)
            nn.ReLU(),
            nn.Conv2d(32, 96, 3, 2, 1),  # →(96,60,80)
            nn.AdaptiveAvgPool2d(1)  # →(96,1,1)
        )

    def forward(self, images, num_rays):
        """
        确保输出形状：(B×V, num_rays, 96)
        """
        BxV = images.shape[0]  # 8
        # 卷积特征提取
        feat = self.conv(images)  # (B×V, 96, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B×V, 96)
        # 扩展到 (B×V, num_rays, 96) → 关键：用repeat_interleave确保维度正确
        feat = feat.unsqueeze(1).repeat_interleave(num_rays, dim=1)  # 替代repeat，避免形状异常
        return feat


class SO3RotInvPointTransformer(nn.Module):
    """SO(3)旋转不变Transformer：利用相对方向（SO(3)不变量），兼容齐次射线"""

    def __init__(self, in_dim=128, num_heads=1, local_radius=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.local_radius = local_radius
        self.rel_dir_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_heads)
        )
        self.value_proj = nn.Linear(in_dim, in_dim)
        self.output_proj = nn.Linear(in_dim, in_dim)
        # 新增：特征区分投影层
        self.distinct_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )

    def forward(self, x, ray_3d_coords, ray_dirs):
        """
        Args:
            x: 联合特征 (B×V, N, 128)
            ray_3d_coords: 旋转后3D坐标 (B×V, N, 3)
            ray_dirs: 旋转后齐次射线方向 (B×V, N, 3) → 新增，用于计算相对方向
        Returns:
            latent_vecs: SO(3)旋转不变隐向量 (B×V, N, 128)
        """
        B, N, _ = x.shape

        # 1. 局部邻域掩码（3D距离<local_radius，与旋转无关）
        neighbor_mask = torch.cdist(ray_3d_coords, ray_3d_coords) < self.local_radius
        neighbor_mask = neighbor_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        rel_dirs = ray_dirs.unsqueeze(2) - ray_dirs.unsqueeze(1)
        rel_dirs = torch.nn.functional.normalize(rel_dirs, dim=-1)
        attn_weights = self.rel_dir_mlp(rel_dirs)
        attn_weights = attn_weights.permute(0, 3, 1, 2)
        min_val = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~neighbor_mask, min_val)
        attn_weights = F.softmax(attn_weights, dim=-1)

        v = self.value_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).reshape(B, N, 128)
        out = x + self.output_proj(out)

        # 新增：强化特征区分度
        out = self.distinct_proj(out)
        return out


class MinimalAnchorHead(nn.Module):
    """基础锚框匹配头：输出4个锚框匹配度（修复输出维度）"""

    def __init__(self, latent_dim=128, num_anchors=4):
        super().__init__()
        self.num_anchors = num_anchors
        # 关键修复：MLP最后一层输出维度设为num_anchors（4），确保输出3维
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_anchors)  # 输出4个值（对应4个锚框）
        )

    def forward(self, latent_vecs):
        """
        输入：latent_vecs (B×V, N, 128) → 8,1024,128
        输出：match_scores (B×V, N, 4) → 8,1024,4
        """
        match_scores = self.mlp(latent_vecs)  # 直接输出3维，无需squeeze
        return torch.sigmoid(match_scores)


class MinimalRegHead(nn.Module):
    """基础尺度回归头：预测偏移量"""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, latent_vecs):
        return self.mlp(latent_vecs).squeeze(-1)  # (B,N)


# 整体模型修改：forward中传入ray_dirs到Transformer
class MinimalDroneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.geom_mlp = MinimalGeomMLP()
        self.appear_backbone = MinimalAppearBackbone()
        # 替换为SO(3)旋转不变Transformer
        self.transformer = SO3RotInvPointTransformer(in_dim=128, num_heads=2)
        self.anchor_head = MinimalAnchorHead()
        self.reg_head = MinimalRegHead()

    def forward(self, data_dict):
        rays = data_dict["rays"]  # (B×V, N, 3) → 8,1024,3
        images = data_dict["images"]  # (B×V, 3, 480, 640) →8,3,480,640
        ray_3d_coords = data_dict["ray_3d_coords"]  # (B×V, N, 3)→8,1024,3
        anchor_centers = data_dict["anchor_centers"]  # (4,)
        N = rays.shape[1]  # 1024
        BxV = rays.shape[0]  # 8

        # 1. 双分支特征提取（添加形状校验）
        geom_feat = self.geom_mlp(rays)  # (B×V, N, 32)→8,1024,32
        assert geom_feat.shape == (BxV, N, 32), f"geom_feat形状错误：{geom_feat.shape}"

        appear_feat = self.appear_backbone(images, num_rays=N)  # (B×V, N, 96)→8,1024,96
        assert appear_feat.shape == (BxV, N, 96), f"appear_feat形状错误：{appear_feat.shape}"

        joint_feat = torch.cat([geom_feat, appear_feat], dim=-1)  # (8,1024,128)
        assert joint_feat.shape == (BxV, N, 128), f"joint_feat形状错误：{joint_feat.shape}"

        # 2. Transformer编码（添加形状校验）
        latent_vecs = self.transformer(joint_feat, ray_3d_coords, rays)  # (8,1024,128)
        assert latent_vecs.shape == (BxV, N, 128), f"latent_vecs形状错误：{latent_vecs.shape}"

        # 3. 锚框匹配+尺度回归（修复维度）
        match_scores = self.anchor_head(latent_vecs)  # (8,1024,4)
        assert match_scores.shape == (BxV, N, 4), f"match_scores形状错误：{match_scores.shape}"

        delta_t = self.reg_head(latent_vecs)  # (8,1024)
        topk_anchor_idx = match_scores.argmax(dim=-1)  # (8,1024) → 正确2维

        # 强制3维扩展
        anchor_centers_expand = anchor_centers.unsqueeze(0).unsqueeze(0).repeat(BxV, N, 1)  # (8,1024,4)
        topk_anchor_idx_3d = topk_anchor_idx.unsqueeze(-1)  # (8,1024,1) → 正确3维

        # 打印最终维度校验（运行一次后可删除）
        print(f"anchor_centers_expand: {anchor_centers_expand.shape}")  # 8,1024,4
        print(f"topk_anchor_idx_3d: {topk_anchor_idx_3d.shape}")  # 8,1024,1

        # 最终gather（维度完全匹配）
        topk_centers = torch.gather(
            anchor_centers_expand,
            dim=2,
            index=topk_anchor_idx_3d
        ).squeeze(-1)  # (8,1024)

        pred_depth = topk_centers + delta_t
        return {
            "latent_vecs": latent_vecs,
            "match_scores": match_scores,
            "pred_depth": pred_depth,
            "topk_anchor_idx": topk_anchor_idx
        }