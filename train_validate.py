import torch
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from model import MinimalDroneModel
from data_generator import generate_synthetic_data
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# 基础配置（直接硬编码，快速修改）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4  # 从8→4，减少一半batch
NUM_RAYS = 1024  # 从2048→1024，减少射线数
LOCAL_RADIUS = 0.6  # 适当扩大局部半径，保证邻域点数
EPOCHS = 40
LR = 3e-5



# 损失函数（仅保留核心2个）
class MinimalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.L1Loss()  # 替换SmoothL1，对大误差更敏感
        self.pos_dist_loss = nn.MSELoss()

    def forward(self, outputs, data_dict):
        # 1. 锚框分类损失（提高权重，让模型优先选对锚框）
        cls_loss = self.cls_criterion(outputs["match_scores"].permute(0, 2, 1), data_dict["gt_anchor_idx"])

        # 2. 尺度回归损失（改用绝对误差，初始阶段更易收敛）
        gt_depth = data_dict["gt_depth"]
        pred_depth = outputs["pred_depth"]
        reg_loss = self.reg_criterion(pred_depth, gt_depth)  # 绝对误差，梯度更稳定

        # 3. 强化3D位置区分损失（增加权重，加大远距惩罚）
        latent_vecs = outputs["latent_vecs"]
        ray_3d_coords = data_dict["ray_3d_coords"]
        BxV, N, _ = latent_vecs.shape
        sample_idx = torch.randint(0, N, (100,), device=latent_vecs.device)
        sample_vecs = latent_vecs[:, sample_idx]
        sample_coords = ray_3d_coords[:, sample_idx]

        coord_dist = torch.cdist(sample_coords, sample_coords)
        vec1 = sample_vecs.unsqueeze(2)
        vec2 = sample_vecs.unsqueeze(1)
        vec_sim = F.cosine_similarity(vec1, vec2, dim=-1)

        # 强化远距惩罚：距离>1m的样本，相似度目标设为0
        target_sim = torch.where(coord_dist < 1.0, 1 - coord_dist, torch.zeros_like(coord_dist))
        pos_loss = self.pos_dist_loss(vec_sim, target_sim)

        # 4. 锚框中心约束（不变）
        topk_anchor_idx = outputs["topk_anchor_idx"]
        anchor_centers = data_dict["anchor_centers"]
        anchor_centers_expand = anchor_centers.unsqueeze(0).unsqueeze(0).repeat(BxV, N, 1)
        topk_centers = torch.gather(anchor_centers_expand, dim=2, index=topk_anchor_idx.unsqueeze(-1)).squeeze(-1)
        anchor_center_loss = self.reg_criterion(topk_centers, gt_depth)

        # 关键调整：权重重新分配（优先保证锚框选对、尺度回归准）
        total_loss = 0.4 * cls_loss + 0.3 * reg_loss + 0.2 * pos_loss + 0.1 * anchor_center_loss
        loss_dict = {
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item(),
            "pos_loss": pos_loss.item(),
            "anchor_center_loss": anchor_center_loss.item()
        }
        return total_loss, loss_dict

# 核心验证指标（判断思路可行性）
def validate_core_metrics(outputs, data_dict):
    latent_vecs = outputs["latent_vecs"]
    ray_3d_coords = data_dict["ray_3d_coords"]
    gt_depth = data_dict["gt_depth"]
    pred_depth = outputs["pred_depth"]
    batch_size = data_dict["batch_size"]
    num_rot_views = data_dict["num_rot_views"]
    N = ray_3d_coords.shape[1]

    metrics = {}

    # 1. 简化3D位置关联性计算（减少循环次数）
    adjacent_sim_list = []
    distant_sim_list = []
    # 只取前2个batch计算，减少显存占用
    for b in range(min(2, latent_vecs.shape[0])):
        # 只采样20个点，而非50个
        sample_idx = np.random.choice(N, 20, replace=False)
        for idx in sample_idx:
            dists = torch.norm(ray_3d_coords[b] - ray_3d_coords[b, idx], dim=-1)
            adjacent_idx = torch.where((dists < LOCAL_RADIUS) & (dists > 0))[0]
            distant_idx = torch.where(dists > 1.0)[0]
            if len(adjacent_idx) > 0:
                adjacent_sim = F.cosine_similarity(latent_vecs[b, idx:idx + 1],
                                                   latent_vecs[b, adjacent_idx]).mean().item()
                adjacent_sim_list.append(adjacent_sim)
            if len(distant_idx) > 0:
                distant_sim = F.cosine_similarity(latent_vecs[b, idx:idx + 1],
                                                  latent_vecs[b, distant_idx]).mean().item()
                distant_sim_list.append(distant_sim)

    metrics["adjacent_sim"] = np.mean(adjacent_sim_list) if adjacent_sim_list else 0.0
    metrics["distant_sim"] = np.mean(distant_sim_list) if distant_sim_list else 1.0

    # 2. 尺度精度（不变）
    rel_error = torch.abs(pred_depth - gt_depth) / gt_depth
    metrics["mean_rel_error"] = rel_error.mean().cpu().item()
    metrics["rel_error_less_5%"] = (rel_error < 0.05).float().mean().cpu().item()

    # 3. 旋转一致性（只计算前1个batch）
    rot_consistency_sim = []
    for b in range(min(1, batch_size)):
        view0_idx = b * num_rot_views
        view1_idx = view0_idx + 1
        if view1_idx >= latent_vecs.shape[0]:
            break
        kdtree = torch.cdist(ray_3d_coords[view1_idx], ray_3d_coords[view0_idx])
        match_idx = torch.argmin(kdtree, dim=-1)
        sim = F.cosine_similarity(
            latent_vecs[view1_idx],
            latent_vecs[view0_idx][match_idx],
            dim=-1
        ).mean().cpu().item()
        rot_consistency_sim.append(sim)

    metrics["rot_consistency_sim"] = np.mean(rot_consistency_sim) if rot_consistency_sim else 0.0
    return metrics


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("警告：未检测到GPU，将使用CPU（速度极慢）！")
        return

    # 1. 初始化模型、损失、优化器（确保模型在GPU）
    model = MinimalDroneModel().to(DEVICE)
    criterion = MinimalLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 2. 初始化混合精度缩放器（关键）
    scaler = GradScaler()  # 自动缩放梯度，避免FP16下溢

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loss_dict_avg = {"cls_loss": 0, "reg_loss": 0}

        # 3. 生成数据（已在GPU，无需转移）
        data_dict = generate_synthetic_data(
            batch_size=BATCH_SIZE,
            num_rays=NUM_RAYS,
            num_rot_views=2
        )
        # 图像也直接在GPU生成
        num_views_total = data_dict["rays"].shape[0]
        data_dict["images"] = torch.randn(num_views_total, 3, 480, 640, device=DEVICE)

        # 4. 混合精度训练（核心加速部分）
        optimizer.zero_grad()
        with autocast():  # 自动将计算转为FP16
            outputs = model(data_dict)
            loss, loss_dict = criterion(outputs, data_dict)

        # 反向传播（使用scaler处理FP16梯度）
        scaler.scale(loss).backward()  # 缩放损失，避免梯度下溢
        scaler.step(optimizer)  # 优化器步骤
        scaler.update()  # 更新缩放器

        # 5. 损失累计（确保在GPU上计算，不转移到CPU）
        total_loss += loss.item()  # item()会自动转移到CPU，但只取一个标量，开销小
        for k in loss_dict_avg.keys():
            loss_dict_avg[k] += loss_dict[k]

        # 6. 验证部分也启用autocast（可选，进一步加速）
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad(), autocast():  # 验证也用FP16
                val_metrics = validate_core_metrics(outputs, data_dict)
            model.train()

            # 打印结果
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(
                f"Loss: Total={total_loss:.4f}, Cls={loss_dict_avg['cls_loss']:.4f}, Reg={loss_dict_avg['reg_loss']:.4f}")
            print(f"核心指标：")
            print(
                f"  3D位置关联性：相邻相似度={val_metrics['adjacent_sim']:.3f}，远距相似度={val_metrics['distant_sim']:.3f}（达标：相邻>0.8，远距<0.3）")
            print(
                f"  尺度预测精度：平均相对误差={val_metrics['mean_rel_error']:.3f}，误差<5%占比={val_metrics['rel_error_less_5%']:.3f}（达标：平均<0.05）")
            print(f"  SO(3)旋转一致性：旋转后隐向量相似度={val_metrics['rot_consistency_sim']:.3f}（达标：>0.9）")  # 新增
            print("-" * 50)

    # 训练结束保存模型（可选）
    torch.save(model.state_dict(), "minimal_model.pth")
    print("最小验证模型保存完成！")


if __name__ == "__main__":
    main()