import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from config.base_config import BaseConfig


class SyntheticDataGenerator:
    def __init__(self):
        self.device = BaseConfig.DEVICE
        self.num_rays = BaseConfig.NUM_RAYS
        self.img_size = (BaseConfig.IMG_HEIGHT, BaseConfig.IMG_WIDTH)
        self.min_depth = BaseConfig.MIN_DEPTH
        self.max_depth = BaseConfig.MAX_DEPTH

    def generate_camera_intrinsic(self):
        """生成相机内参矩阵"""
        fx = fy = 500  # 焦距
        cx, cy = self.img_size[1] / 2, self.img_size[0] / 2  # 主点
        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], device=self.device, dtype=torch.float32)
        return K

    def generate_ray_directions(self, K):
        """生成相机系射线方向"""
        # 随机采样像素坐标
        u = torch.randint(0, self.img_size[1], (self.num_rays,), device=self.device)
        v = torch.randint(0, self.img_size[0], (self.num_rays,), device=self.device)

        # 像素→相机系射线
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        z = torch.ones_like(x, device=self.device)
        ray_dir = torch.stack([x, y, z], dim=-1)
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
        return ray_dir, (u, v)

    def generate_so3_rotation(self):
        """生成随机SO(3)旋转矩阵"""
        rot_euler = np.random.uniform(-30, 30, 3) * np.pi / 180  # ±30°
        R_mat = R.from_euler('xyz', rot_euler).as_matrix()
        return torch.tensor(R_mat, device=self.device, dtype=torch.float32)

    def generate_depth_map(self, ray_dir):
        """生成随机深度图（模拟真实场景）"""
        # 模拟地面+障碍物深度分布
        ground_depth = self.min_depth + (self.max_depth - self.min_depth) * 0.3
        obstacle_depth = torch.randn_like(ray_dir[:, 0], device=self.device) * 2 + 15
        obstacle_mask = torch.rand_like(ray_dir[:, 0], device=self.device) > 0.7

        depth = torch.where(obstacle_mask, obstacle_depth, ground_depth)
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        return depth

    def generate_data(self, num_views=2):
        """生成单批次合成数据（多视角）"""
        K = self.generate_camera_intrinsic()
        data_list = []

        for _ in range(num_views):
            # 生成射线方向
            ray_dir, (u, v) = self.generate_ray_directions(K)

            # 生成随机旋转
            rot_mat = self.generate_so3_rotation()
            ray_dir_rot = torch.matmul(ray_dir, rot_mat.T)  # 旋转射线

            # 生成深度
            depth = self.generate_depth_map(ray_dir)

            # 生成3D点（相机系）
            ray_3d = ray_dir_rot * depth.unsqueeze(-1)

            # 生成随机RGB图像
            img = torch.randn((BaseConfig.CHANNELS, self.img_size[0], self.img_size[1]),
                              device=self.device, dtype=torch.float32)

            data = {
                "image": img,
                "ray_dir": ray_dir_rot,
                "ray_3d": ray_3d,
                "depth": depth,
                "K": K,
                "rot_mat": rot_mat,
                "uv": (u, v)
            }
            data_list.append(data)

        return data_list, K

    def batch_generate(self, batch_size=4):
        """批量生成数据"""
        batch_data = []
        for _ in range(batch_size):
            views_data, K = self.generate_data(num_views=2)
            batch_data.append(views_data)
        return batch_data, K


if __name__ == "__main__":
    # 测试生成
    generator = SyntheticDataGenerator()
    batch_data, K = generator.batch_generate(batch_size=2)
    print(f"生成批次大小: {len(batch_data)}")
    print(f"单样本视角数: {len(batch_data[0])}")
    print(f"射线方向形状: {batch_data[0][0]['ray_dir'].shape}")
    print(f"深度形状: {batch_data[0][0]['depth'].shape}")