import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import BaseConfig


class LossFunctions:
    """损失函数集合（阶段二专用）"""

    def __init__(self):
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.HuberLoss(delta=1.0)  # 适配深度范围，平衡大/小误差
        self.pos_dist_criterion = nn.MSELoss()
        self.anchor_center_criterion = nn.L1Loss()

    def cls_loss(self, match_scores, gt_anchor_idx):
        """锚框分类损失"""
        # match_scores: (B×V, N, num_anchors)
        # gt_anchor_idx: (B×V, N)
        return self.cls_criterion(match_scores.permute(0, 2, 1), gt_anchor_idx)

    def pos_dist_loss(self, latent_vecs, ray_3d):
        """3D位置区分损失（强化远距点差异）"""
        BxV, N, _ = latent_vecs.shape
        device = latent_vecs.device

        # 随机采样100个点计算距离和相似度（控制显存）
        sample_idx = torch.randint(0, N, (100,), device=device)
        sample_vecs = latent_vecs[:, sample_idx]  # (B×V, 100, 128)
        sample_coords = ray_3d[:, sample_idx]  # (B×V, 100, 3)

        # 计算采样点之间的3D距离和隐向量相似度
        coord_dist = torch.cdist(sample_coords, sample_coords)  # (B×V, 100, 100)
        vec1 = sample_vecs.unsqueeze(2)  # (B×V, 100, 1, 128)
        vec2 = sample_vecs.unsqueeze(1)  # (B×V, 1, 100, 128)
        vec_sim = F.cosine_similarity(vec1, vec2, dim=-1)  # (B×V, 100, 100)

        # 目标相似度：近距高、远距低（距离>0.5m强制为0）
        target_sim = torch.where(
            coord_dist < 0.5,
            1 - coord_dist,  # 近距：相似度随距离线性降低
            torch.zeros_like(coord_dist)  # 远距：相似度强制为0，加大惩罚
        )

        return self.pos_dist_criterion(vec_sim, target_sim)

    def reg_loss(self, pred_depth, gt_depth):
        """深度回归损失（HuberLoss）"""
        return self.reg_criterion(pred_depth, gt_depth)

    def anchor_center_loss(self, selected_anchor, gt_depth):
        """锚框中心约束损失（确保锚框中心贴近真值）"""
        return self.anchor_center_criterion(selected_anchor, gt_depth)

    def compute_total_loss(self, pred_dict, data, loss_weights):
        """计算总损失（按权重加权）"""
        # 1. 构建GT锚框索引（根据真值深度匹配最近的锚框）
        gt_depth = data["depth"]  # (B×V, N)
        anchor_centers = BaseConfig.ANCHOR_CENTERS.to(gt_depth.device)
        gt_anchor_idx = torch.argmin(
            torch.abs(gt_depth.unsqueeze(-1) - anchor_centers.unsqueeze(0).unsqueeze(0)),
            dim=-1
        )  # (B×V, N)

        # 2. 计算各分项损失
        cls_loss_val = self.cls_loss(pred_dict["match_scores"], gt_anchor_idx)

        pos_loss_val = self.pos_dist_loss(pred_dict["latent_vecs"], data["ray_3d"])

        reg_loss_val = self.reg_loss(pred_dict["pred_depth"], gt_depth)

        # 计算选中的锚框中心
        anchor_centers_expand = anchor_centers.unsqueeze(0).unsqueeze(0).repeat(
            gt_depth.shape[0], gt_depth.shape[1], 1
        )  # (B×V, N, num_anchors)
        selected_anchor = torch.gather(
            anchor_centers_expand,
            dim=2,
            index=pred_dict["anchor_idx"].unsqueeze(-1)
        ).squeeze(-1)  # (B×V, N)
        anchor_center_loss_val = self.anchor_center_loss(selected_anchor, gt_depth)

        # 3. 总损失加权
        total_loss = (
                loss_weights["cls_loss"] * cls_loss_val +
                loss_weights["pos_loss"] * pos_loss_val +
                loss_weights["reg_loss"] * reg_loss_val +
                loss_weights["anchor_center_loss"] * anchor_center_loss_val
        )

        # 4. 返回损失字典
        loss_dict = {
            "total_loss": total_loss,
            "cls_loss": cls_loss_val,
            "pos_loss": pos_loss_val,
            "reg_loss": reg_loss_val,
            "anchor_center_loss": anchor_center_loss_val
        }
        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试损失函数
    loss_fn = LossFunctions()
    # 构造测试数据
    pred_dict = {
        "match_scores": torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.ANCHOR_NUM), device=BaseConfig.DEVICE),
        "latent_vecs": torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.LATENT_DIM), device=BaseConfig.DEVICE),
        "pred_depth": torch.randn((8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE) * 2 + 15,
        "anchor_idx": torch.randint(0, BaseConfig.ANCHOR_NUM, (8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE)
    }
    data = {
        "depth": torch.randn((8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE) * 2 + 15,
        "ray_3d": torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    }
    loss_weights = {
        "cls_loss": 0.2,
        "pos_loss": 0.3,
        "reg_loss": 0.4,
        "anchor_center_loss": 0.1
    }
    total_loss, loss_dict = loss_fn.compute_total_loss(pred_dict, data, loss_weights)
    print("损失计算测试:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")