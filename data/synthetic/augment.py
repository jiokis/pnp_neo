import torch
import torchvision.transforms as transforms
from config.data_config import DataConfig


class SyntheticDataAugmentor:
    def __init__(self):
        self.augment_config = DataConfig.AUGMENT_CONFIG
        self.transform_list = self._build_transforms()

    def _build_transforms(self):
        """构建数据增强变换"""
        transform_list = []

        # 随机旋转
        if self.augment_config["random_rotate"]:
            angle_range = self.augment_config["rotate_range"]
            transform_list.append(transforms.RandomRotation(degrees=angle_range))

        # 亮度调整
        if self.augment_config["brightness_jitter"]:
            brightness = self.augment_config["brightness_range"]
            transform_list.append(transforms.ColorJitter(brightness=brightness))

        # 归一化
        if self.augment_config["normalize"]:
            mean = self.augment_config["mean"]
            std = self.augment_config["std"]
            transform_list.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(transform_list)

    def augment_image(self, image):
        """图像增强"""
        # image shape: (C, H, W)
        return self.transform_list(image)

    def augment_ray(self, ray_dir, rot_angle=None):
        """射线随机旋转增强"""
        if not self.augment_config["random_rotate"] or rot_angle is None:
            return ray_dir

        # 绕Z轴旋转射线
        angle_rad = torch.tensor(rot_angle * torch.pi / 180, device=ray_dir.device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        rot_mat = torch.tensor([[cos_a, -sin_a, 0],
                                [sin_a, cos_a, 0],
                                [0, 0, 1]], device=ray_dir.device)
        ray_dir_rot = torch.matmul(ray_dir, rot_mat)
        return ray_dir_rot

    def __call__(self, data):
        """对单视角数据执行增强"""
        # 图像增强
        data["image"] = self.augment_image(data["image"])

        # 射线增强（随机旋转）
        if self.augment_config["random_rotate"]:
            rot_angle = torch.randint(*self.augment_config["rotate_range"], (1,)).item()
            data["ray_dir"] = self.augment_ray(data["ray_dir"], rot_angle)
            data["ray_3d"] = self.augment_ray(data["ray_3d"], rot_angle)

        return data