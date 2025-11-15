import torch
import torch.nn as nn
from config.base_config import BaseConfig
from model.backbone.geom_branch import GeometricBranch
from model.backbone.appearance_branch import MobileNetV3AppearanceBranch
from model.transformer.so3_rot_inv_transformer import SO3RotInvTransformer
from model.head.anchor_head import AnchorHead
from model.head.depth_reg_head import DepthRegressionHead
from model.auxiliary.multi_scale_fusion import MultiScaleFusion
from model.auxiliary.depth_calib import DepthCalibration


class DroneCrossViewModel(nn.Module):
    """完整无人机跨视角匹配模型"""

    def __init__(self):
        super().__init__()
        # 骨干网络
        self.geom_branch = GeometricBranch()
        self.appear_branch = MobileNetV3AppearanceBranch()

        # 特征融合
        self.feat_fusion = nn.Linear(32 + 96, BaseConfig.LATENT_DIM)

        # 多尺度融合
        self.multi_scale_fusion = MultiScaleFusion(input_dim=BaseConfig.LATENT_DIM)

        # 旋转不变Transformer
        self.transformer = SO3RotInvTransformer(
            d_model=BaseConfig.LATENT_DIM,
            nhead=BaseConfig.TRANSFORMER_HEADS,
            local_radius=BaseConfig.LOCAL_RADIUS
        )

        # 预测头
        self.anchor_head = AnchorHead(input_dim=BaseConfig.LATENT_DIM)
        self.depth_reg_head = DepthRegressionHead(
            # input_dim=BaseConfig.LATENT_DIM,
            input_dim=128,  # 与latent_vecs的特征维度一致（根据你的模型调整，通常是256/512）
            num_anchors=16,

        )

        # 深度校准
        self.depth_calib = DepthCalibration()

    def forward(self, data):
        """
        输入: data - 包含单批次多视角数据
              data["ray_dir"]: (B×V, N, 3)
              data["ray_3d"]: (B×V, N, 3)
              data["image"]: (B×V, C, H, W)
        输出: pred_dict - 预测结果字典
        """
        ray_dir = data["ray_dir"]
        ray_3d = data["ray_3d"]
        image = data["image"]
        BxV, N = ray_dir.shape[:2]

        # 1. 特征提取
        geom_feat = self.geom_branch(ray_dir)  # (B×V, N, 32)
        appear_feat = self.appear_branch(image, num_rays=N)  # (B×V, N, 96)

        # 2. 特征融合
        fused_feat = torch.cat([geom_feat, appear_feat], dim=-1)  # (B×V, N, 128)
        fused_feat = self.feat_fusion(fused_feat)  # (B×V, N, 128)

        # 3. 多尺度融合
        multi_scale_feat = self.multi_scale_fusion(fused_feat)  # (B×V, N, 128)

        # 4. Transformer编码
        latent_vecs = self.transformer(multi_scale_feat, ray_3d, ray_dir)  # (B×V, N, 128)

        # 5. 锚框匹配
        match_scores, conf_scores = self.anchor_head(latent_vecs)  # (B×V, N, 4), (B×V, N, 1)
        anchor_idx = torch.argmax(match_scores, dim=-1)  # (B×V, N)

        # 6. 深度回归
        pred_depth, residual_depth = self.depth_reg_head(latent_vecs, anchor_idx)  # (B×V, N)

        # 7. 深度校准（训练阶段不启用，推理阶段启用）
        if not self.training:
            pred_depth = self.depth_calib(pred_depth, ray_3d)

        # 整理输出
        pred_dict = {
            "latent_vecs": latent_vecs,
            "match_scores": match_scores,
            "conf_scores": conf_scores,
            "anchor_idx": anchor_idx,
            "pred_depth": pred_depth,
            "residual_depth": residual_depth
        }
        return pred_dict


if __name__ == "__main__":
    # 测试完整模型
    model = DroneCrossViewModel().to(BaseConfig.DEVICE)
    # 构造输入数据
    input_data = {
        "ray_dir": torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE),
        "ray_3d": torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE),
        "image": torch.randn((8, 3, BaseConfig.IMG_HEIGHT, BaseConfig.IMG_WIDTH), device=BaseConfig.DEVICE)
    }
    pred_dict = model(input_data)
    print("模型输出键值:")
    for k, v in pred_dict.items():
        print(f"  {k}: {v.shape}")