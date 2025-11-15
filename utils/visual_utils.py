import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
from config.base_config import BaseConfig

class VisualUtils:
    """可视化工具类"""
    def __init__(self):
        self.device = BaseConfig.DEVICE
        self.img_size = (BaseConfig.IMG_WIDTH, BaseConfig.IMG_HEIGHT)

    def visualize_depth_map(self, pred_depth, gt_depth, save_path="depth_visualize.png"):
        """可视化深度图对比"""
        # 转换为numpy数组
        pred_depth_np = pred_depth.cpu().detach().numpy()
        gt_depth_np = gt_depth.cpu().detach().numpy()

        # 归一化到[0, 255]
        pred_depth_norm = self._normalize_depth(pred_depth_np)
        gt_depth_norm = self._normalize_depth(gt_depth_np)

        # 转为彩色图
        pred_depth_color = cv2.applyColorMap(pred_depth_norm, cv2.COLORMAP_JET)
        gt_depth_color = cv2.applyColorMap(gt_depth_norm, cv2.COLORMAP_JET)

        # 拼接显示
        combined = np.hstack([gt_depth_color, pred_depth_color])
        cv2.imwrite(save_path, combined)
        print(f"深度图对比已保存至: {save_path}")

    def _normalize_depth(self, depth):
        """深度图归一化"""
        depth = np.clip(depth, BaseConfig.MIN_DEPTH, BaseConfig.MAX_DEPTH)
        depth_norm = ((depth - BaseConfig.MIN_DEPTH) / (BaseConfig.MAX_DEPTH - BaseConfig.MIN_DEPTH) * 255).astype(np.uint8)
        return depth_norm

    def visualize_latent_space(self, latent_vecs, ray_3d, save_path="latent_tsne.png"):
        """TSNE可视化隐空间"""
        # 随机采样1000个点（避免计算量过大）
        N = latent_vecs.shape[1]
        sample_idx = np.random.choice(N, min(1000, N), replace=False)
        latent_sample = latent_vecs[0, sample_idx].cpu().detach().numpy()  # 取第一个样本
        ray_3d_sample = ray_3d[0, sample_idx].cpu().detach().numpy()

        # TSNE降维
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_sample)

        # 按深度着色
        depth = np.linalg.norm(ray_3d_sample, axis=1)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=depth, cmap='viridis', s=10)
        plt.colorbar(scatter, label='Depth (m)')
        plt.title('Latent Space TSNE Visualization (Colored by Depth)')
        plt.xlabel('TSNE Dimension 1')
        plt.ylabel('TSNE Dimension 2')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"隐空间可视化已保存至: {save_path}")

    def visualize_cross_view_matching(self, img0, img1, match_idx, uv0, uv1, save_path="match_visualize.png"):
        """可视化跨视角匹配点对"""
        # 转换图像格式 (C, H, W) → (H, W, C)
        img0_np = img0.cpu().detach().numpy().transpose(1, 2, 0)
        img1_np = img1.cpu().detach().numpy().transpose(1, 2, 0)

        # 归一化到[0, 255]
        img0_np = self._normalize_image(img0_np)
        img1_np = self._normalize_image(img1_np)

        # 转换为BGR格式（适配OpenCV）
        img0_bgr = cv2.cvtColor(img0_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img1_bgr = cv2.cvtColor(img1_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 随机选择100个匹配点绘制
        uv0_np = uv0.cpu().detach().numpy()
        uv1_np = uv1.cpu().detach().numpy()
        match_idx_np = match_idx.cpu().detach().numpy()

        sample_match_idx = np.random.choice(len(uv0_np), min(100, len(uv0_np)), replace=False)
        for i in sample_match_idx:
            # 绘制点
            cv2.circle(img0_bgr, (int(uv0_np[0][i]), int(uv0_np[1][i])), 3, (0, 255, 0), -1)
            cv2.circle(img1_bgr, (int(uv1_np[0][match_idx_np[i]]), int(uv1_np[1][match_idx_np[i]])), 3, (0, 255, 0), -1)

        # 拼接图像
        combined = np.hstack([img0_bgr, img1_bgr])
        cv2.imwrite(save_path, combined)
        print(f"跨视角匹配可视化已保存至: {save_path}")

    def _normalize_image(self, img):
        """图像归一化"""
        img = img - img.min()
        img = img / img.max() * 255
        return img

if __name__ == "__main__":
    # 测试可视化工具
    visualizer = VisualUtils()
    # 构造测试数据
    pred_depth = torch.randn((8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE) * 2 + 15
    gt_depth = torch.randn((8, BaseConfig.NUM_RAYS), device=BaseConfig.DEVICE) * 2 + 15
    latent_vecs = torch.randn((8, BaseConfig.NUM_RAYS, BaseConfig.LATENT_DIM), device=BaseConfig.DEVICE)
    ray_3d = torch.randn((8, BaseConfig.NUM_RAYS, 3), device=BaseConfig.DEVICE)
    img0 = torch.randn((3, BaseConfig.IMG_HEIGHT, BaseConfig.IMG_WIDTH), device=BaseConfig.DEVICE)
    img1 = torch.randn((3, BaseConfig.IMG_HEIGHT, BaseConfig.IMG_WIDTH), device=BaseConfig.DEVICE)
    uv0 = (torch.randint(0, BaseConfig.IMG_WIDTH, (BaseConfig.NUM_RAYS,), device=BaseConfig.DEVICE),
           torch.randint(0, BaseConfig.IMG_HEIGHT, (BaseConfig.NUM_RAYS,), device=BaseConfig.DEVICE))
    uv1 = (torch.randint(0, BaseConfig.IMG_WIDTH, (BaseConfig.NUM_RAYS,), device=BaseConfig.DEVICE),
           torch.randint(0, BaseConfig.IMG_HEIGHT, (BaseConfig.NUM_RAYS,), device=BaseConfig.DEVICE))
    match_idx = torch.randint(0, BaseConfig.NUM_RAYS, (BaseConfig.NUM_RAYS,), device=BaseConfig.DEVICE)

    # 测试可视化
    visualizer.visualize_depth_map(pred_depth[0], gt_depth[0], "test_depth.png")
    visualizer.visualize_latent_space(latent_vecs, ray_3d, "test_latent.png")
    visualizer.visualize_cross_view_matching(img0, img1, match_idx, uv0, uv1, "test_match.png")