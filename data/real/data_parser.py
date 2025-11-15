import os
import torch
import numpy as np
from PIL import Image
from config.base_config import BaseConfig
from config.data_config import DataConfig


class EurocParser:
    """EuRoC数据集解析器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.cam0_path = os.path.join(data_path, "mav0", "cam0", "data")
        self.depth_path = os.path.join(data_path, "mav0", "depth0", "data")  # 深度图路径
        self.state_path = os.path.join(data_path, "mav0", "state_groundtruth_estimate0", "data.csv")

        # 加载相机内参
        self.K = self._load_calibration()
        # 加载轨迹数据
        self.trajectory = self._load_trajectory()

    def _load_calibration(self):
        """加载相机内参"""
        calib_path = os.path.join(self.data_path, "mav0", "cam0", "sensor.yaml")
        import yaml
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)
        fx = calib["intrinsics"][0]
        fy = calib["intrinsics"][1]
        cx = calib["intrinsics"][2]
        cy = calib["intrinsics"][3]

        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], device=BaseConfig.DEVICE, dtype=torch.float32)
        return K

    def _load_trajectory(self):
        """加载轨迹数据（时间戳、位置、姿态）"""
        trajectory = {}
        with open(self.state_path, 'r') as f:
            lines = f.readlines()[1:]  # 跳过表头
            for line in lines:
                parts = line.strip().split(',')
                ts = float(parts[0])
                # 位置
                pos = torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])],
                                   device=BaseConfig.DEVICE)
                # 姿态（四元数→旋转矩阵）
                q = torch.tensor([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                                 device=BaseConfig.DEVICE)
                rot_mat = self._quat_to_rot(q)
                trajectory[ts] = {"pos": pos, "rot_mat": rot_mat}
        return trajectory

    @staticmethod
    def _quat_to_rot(quat):
        """四元数转旋转矩阵"""
        qw, qx, qy, qz = quat
        rot_mat = torch.tensor([
            [1 - 2 * qy^2-2 * qz^2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx^2-2 * qz^2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx^2-2 * qy^2]
        ], device = quat.device, dtype = quat.dtype)
        return rot_mat

    def _get_closest_ts(self, ts):
        """找到最接近的轨迹时间戳"""
        ts_list = sorted(self.trajectory.keys())
        idx = np.searchsorted(ts_list, ts)
        if idx == 0:
            return ts_list[0]
        elif idx == len(ts_list):
            return ts_list[-1]
        else:
            return ts_list[idx - 1] if (ts - ts_list[idx - 1]) < (ts_list[idx] - ts) else ts_list[idx]

    def parse_image(self, img_name):
        """解析单张图像"""
        img_path = os.path.join(self.cam0_path, img_name)
        img = Image.open(img_path).convert('RGB')
        # 转为tensor并调整维度 (H,W,C)→(C,H,W)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_tensor.to(BaseConfig.DEVICE)

    def parse_depth(self, depth_name):
        """解析深度图"""
        depth_path = os.path.join(self.depth_path, depth_name)
        depth = Image.open(depth_path)
        depth_tensor = torch.from_numpy(np.array(depth)).float() / 1000.0  # 单位转换为米
        return depth_tensor.to(BaseConfig.DEVICE)

    def generate_rays(self, img_tensor, depth_tensor):
        """生成射线和3D点"""
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        # 随机采样射线
        indices = torch.randperm(H * W, device=BaseConfig.DEVICE)[:BaseConfig.NUM_RAYS]
        u = indices % W
        v = indices // W

        # 生成射线方向
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        z = torch.ones_like(x, device=BaseConfig.DEVICE)
        ray_dir = torch.stack([x, y, z], dim=-1)
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)

        # 采样深度
        depth = depth_tensor[v, u]
        # 过滤无效深度
        valid_mask = (depth >= DataConfig.PREPROCESS_CONFIG["min_depth"]) & \
                     (depth <= DataConfig.PREPROCESS_CONFIG["max_depth"])
        ray_dir = ray_dir[valid_mask]
        depth = depth[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]

        # 生成3D点
        ray_3d = ray_dir * depth.unsqueeze(-1)

        return ray_dir, ray_3d, depth, (u, v)

    def parse_data(self, img_name, depth_name):
        """解析单视角完整数据"""
        # 解析图像和深度
        img_tensor = self.parse_image(img_name)
        depth_tensor = self.parse_depth(depth_name)

        # 生成射线和3D点
        ray_dir, ray_3d, depth, (u, v) = self.generate_rays(img_tensor, depth_tensor)

        # 获取轨迹信息
        ts = float(img_name.split('.')[0])
        closest_ts = self._get_closest_ts(ts)
        traj_data = self.trajectory[closest_ts]

        return {
            "image": img_tensor,
            "ray_dir": ray_dir,
            "ray_3d": ray_3d,
            "depth": depth,
            "K": self.K,
            "rot_mat": traj_data["rot_mat"],
            "pos": traj_data["pos"],
            "uv": (u, v),
            "timestamp": ts
        }


class TumParser(EurocParser):
    """TUM数据集解析器（继承EuRoC解析器，适配路径）"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.cam0_path = os.path.join(data_path, "rgb")
        self.depth_path = os.path.join(data_path, "depth")
        self.state_path = os.path.join(data_path, "groundtruth.txt")

        # 加载相机内参
        self.K = self._load_calibration()
        # 加载轨迹数据
        self.trajectory = self._load_trajectory()

    def _load_calibration(self):
        """加载TUM相机内参"""
        calib_path = os.path.join(self.data_path, "camera_intrinsic.txt")
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            fx = float(lines[0].split()[0])
            fy = float(lines[1].split()[0])
            cx = float(lines[2].split()[0])
            cy = float(lines[3].split()[0])

        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], device=BaseConfig.DEVICE, dtype=torch.float32)
        return K

    def _load_trajectory(self):
        """加载TUM轨迹数据"""
        trajectory = {}
        with open(self.state_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                ts = float(parts[0])
                # 位置
                pos = torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])],
                                   device=BaseConfig.DEVICE)
                # 姿态（四元数→旋转矩阵）
                q = torch.tensor([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                                 device=BaseConfig.DEVICE)
                rot_mat = self._quat_to_rot(q)
                trajectory[ts] = {"pos": pos, "rot_mat": rot_mat}
        return trajectory


# 数据集工厂类
class RealDataParserFactory:
    @staticmethod
    def get_parser(dataset_name, data_path):
        if dataset_name == "euroc":
            return EurocParser(data_path)
        elif dataset_name == "tum":
            return TumParser(data_path)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")


if __name__ == "__main__":
    # 测试解析
    parser = RealDataParserFactory.get_parser("euroc", DataConfig.REAL_DATASETS["euroc"])
    img_names = sorted(os.listdir(parser.cam0_path))[:2]
    depth_names = sorted(os.listdir(parser.depth_path))[:2]
    for img_name, depth_name in zip(img_names, depth_names):
        data = parser.parse_data(img_name, depth_name)
        print(f"图像形状: {data['image'].shape}")
        print(f"射线方向形状: {data['ray_dir'].shape}")
        print(f"深度形状: {data['depth'].shape}")