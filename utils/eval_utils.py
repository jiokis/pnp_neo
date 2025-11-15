import torch
import torch.nn.functional as F
from config.base_config import BaseConfig


class EvaluationMetrics:
    """评估指标计算工具（完整修复版）"""

    def __init__(self):
        self.min_depth = BaseConfig.MIN_DEPTH
        self.max_depth = BaseConfig.MAX_DEPTH
        self.local_radius = BaseConfig.LOCAL_RADIUS
        self.num_rays = BaseConfig.NUM_RAYS  # 射线数（512）
        self.latent_dim = BaseConfig.LATENT_DIM  # 特征维度（128）

    def generate_match_indices(self, vec0, vec1):
        """生成跨视角匹配索引（最近邻匹配，确保形状[B, N]）"""
        B, N, D = vec0.shape
        # 计算vec0与vec1的余弦相似度（[B, N, N]）
        similarity = F.cosine_similarity(vec0.unsqueeze(2), vec1.unsqueeze(1), dim=-1)
        # 每个射线匹配最相似的射线，返回[B, N]形状索引
        match_idx = torch.argmax(similarity, dim=-1)  # 形状：[B, N]
        return match_idx

    def compute_cross_view_match(self, vec0, vec1, ray3d_0, ray3d_1, K):
        """计算跨视角匹配指标（修复参数和维度）"""
        # 确保输入向量形状正确：[B, N, D]
        B, N, D = vec0.shape
        assert vec1.shape == (B, N, D), f"vec1形状错误：期望({B}, {N}, {D})，实际{vec1.shape}"

        # 1. 生成匹配索引（[B, N]）
        match_idx = self.generate_match_indices(vec0, vec1)

        # 2. 提取匹配的vec1（修复gather维度）
        matched_vec1 = torch.gather(
            vec1,
            dim=1,
            index=match_idx.unsqueeze(-1).expand(-1, -1, D)  # 扩展为[B, N, D]
        )

        # 3. 计算旋转一致性
        rot_metrics = self.compute_rotation_consistency(vec0, matched_vec1)
        return rot_metrics

    def compute_rotation_consistency(self, vec0, vec1_matched):
        """计算旋转一致性（确保维度匹配）"""
        # 验证输入形状：[B, N, D]
        assert vec0.shape == vec1_matched.shape, f"vec0与vec1_matched形状不匹配：{vec0.shape} vs {vec1_matched.shape}"
        # 计算余弦相似度（dim=-1按特征维度计算）
        rot_sim = F.cosine_similarity(vec0, vec1_matched, dim=-1).mean().item()
        return {"rot_consistency": rot_sim}

    def compute_position_metrics(self, latent_vecs, ray_3d):
        """计算3D位置关联性指标（修复维度适配）"""
        # latent_vecs形状：[B×V, N, D]，ray_3d形状：[B×V, N, 3]
        BxV, N, _ = latent_vecs.shape
        device = latent_vecs.device

        # 随机采样50个点（确保采样索引不超出范围）
        sample_size = min(50, N)
        sample_idx = torch.randint(0, N, (sample_size,), device=device)
        sample_vecs = latent_vecs[:, sample_idx]  # [B×V, 50, D]
        sample_coords = ray_3d[:, sample_idx]  # [B×V, 50, 3]

        # 计算距离和相似度
        coord_dist = torch.cdist(sample_coords, sample_coords)  # [B×V, 50, 50]
        vec_sim = F.cosine_similarity(sample_vecs.unsqueeze(2), sample_vecs.unsqueeze(1), dim=-1)  # [B×V, 50, 50]

        # 相邻/远距相似度（按批次平均）
        adjacent_sim = []
        distant_sim = []
        for i in range(BxV):
            adj_mask = coord_dist[i] < self.local_radius
            if adj_mask.any():
                adjacent_sim.append(vec_sim[i][adj_mask].mean().item())
            else:
                adjacent_sim.append(0.0)

            dist_mask = coord_dist[i] > 1.0
            if dist_mask.any():
                distant_sim.append(vec_sim[i][dist_mask].mean().item())
            else:
                distant_sim.append(0.0)

        return {
            "adjacent_similarity": sum(adjacent_sim) / BxV,
            "distant_similarity": sum(distant_sim) / BxV
        }

    def compute_scale_metrics(self, pred_depth, gt_depth):
        """补全缺失的尺度预测指标计算"""
        # 过滤无效深度值（超出范围的深度不计入）
        valid_mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
        if not valid_mask.any():
            return {"depth_mae": 0.0, "depth_rmse": 0.0}

        # 计算MAE和RMSE
        pred_depth_valid = pred_depth[valid_mask]
        gt_depth_valid = gt_depth[valid_mask]
        abs_error = torch.abs(pred_depth_valid - gt_depth_valid)
        mae = abs_error.mean().item()
        rmse = torch.sqrt(torch.square(abs_error).mean()).item()

        return {"depth_mae": mae, "depth_rmse": rmse}

    def compute_metrics(self, pred_dict, gt_data, view1_data=None, pred_dict_view1=None):
        """综合计算所有指标（修复参数传递错误）"""
        metrics = {}

        # 1. 尺度预测指标（pred_depth: [B×V, N]，gt_depth: [B×V, N]）
        scale_metrics = self.compute_scale_metrics(pred_dict["pred_depth"], gt_data["depth"])
        metrics.update(scale_metrics)

        # 2. 3D位置关联性指标（latent_vecs: [B×V, N, D]，ray_3d: [B×V, N, 3]）
        pos_metrics = self.compute_position_metrics(pred_dict["latent_vecs"], gt_data["ray_3d"])
        metrics.update(pos_metrics)

        # 3. 跨视角旋转一致性（验证阶段，修复参数传递）
        if view1_data is not None and pred_dict_view1 is not None:
            # 提取latent_vecs张量（之前错误传递了整个字典）
            vec0 = pred_dict["latent_vecs"]  # [B×V, N, D]
            vec1 = pred_dict_view1["latent_vecs"]  # [B×V, N, D]
            ray3d_0 = gt_data["ray_3d"]  # [B×V, N, 3]
            ray3d_1 = view1_data["ray_3d"]  # [B×V, N, 3]
            K = gt_data.get("K", torch.eye(3).to(vec0.device))  # 相机内参

            # 转换为[B, N, D]形状（B×V拆分为B和V，假设V=2视角）
            B = vec0.shape[0] // 2
            vec0 = vec0.reshape(B, 2, self.num_rays, self.latent_dim)[:, 0, :, :]  # 取第一个视角
            vec1 = vec1.reshape(B, 2, self.num_rays, self.latent_dim)[:, 1, :, :]  # 取第二个视角
            ray3d_0 = ray3d_0.reshape(B, 2, self.num_rays, 3)[:, 0, :, :]
            ray3d_1 = ray3d_1.reshape(B, 2, self.num_rays, 3)[:, 1, :, :]

            # 计算跨视角指标
            rot_metrics = self.compute_cross_view_match(vec0, vec1, ray3d_0, ray3d_1, K)
            metrics.update(rot_metrics)

        return metrics


if __name__ == "__main__":
    # 测试评估指标（适配修复后的代码）
    BaseConfig.NUM_RAYS = 512
    BaseConfig.LATENT_DIM = 128
    BaseConfig.MIN_DEPTH = 3.0
    BaseConfig.MAX_DEPTH = 30.0
    BaseConfig.LOCAL_RADIUS = 0.5
    BaseConfig.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_metrics = EvaluationMetrics()
    # 构造测试数据（[B×V=8, N=512, D=128]）
    pred_dict = {
        "pred_depth": torch.randn((8, 512), device=BaseConfig.DEVICE) * 2 + 15,
        "latent_vecs": torch.randn((8, 512, 128), device=BaseConfig.DEVICE)
    }
    gt_data = {
        "depth": torch.randn((8, 512), device=BaseConfig.DEVICE) * 2 + 15,
        "ray_3d": torch.randn((8, 512, 3), device=BaseConfig.DEVICE),
        "K": torch.eye(3, device=BaseConfig.DEVICE)
    }
    view1_data = {
        "ray_3d": torch.randn((8, 512, 3), device=BaseConfig.DEVICE)
    }
    pred_dict_view1 = {
        "latent_vecs": torch.randn((8, 512, 128), device=BaseConfig.DEVICE),
        "pred_depth": torch.randn((8, 512), device=BaseConfig.DEVICE) * 2 + 15
    }

    metrics = eval_metrics.compute_metrics(pred_dict, gt_data, view1_data, pred_dict_view1)
    print("评估指标测试:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")