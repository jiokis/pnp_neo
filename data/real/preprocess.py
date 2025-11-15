import torch
import torchvision.transforms as transforms
from config.base_config import BaseConfig
from config.data_config import DataConfig
import os


class RealDataPreprocessor:
    def __init__(self):
        self.preprocess_config = DataConfig.PREPROCESS_CONFIG
        self.resize = transforms.Resize(self.preprocess_config["resize"])

    def undistort_image(self, image, K, dist_coeffs):
        """图像去畸变（简化版）"""
        if not self.preprocess_config["undistort"] or dist_coeffs is None:
            return image
        # 实际项目中需使用OpenCV的undistort函数
        # 此处简化处理，直接返回原图
        return image

    def resize_data(self, image, depth):
        """调整图像和深度图尺寸"""
        # 调整图像尺寸
        image_resized = self.resize(image)

        # 调整深度图尺寸（使用双线性插值）
        depth_resized = transforms.Resize(self.preprocess_config["resize"],
                                          interpolation=transforms.InterpolationMode.BILINEAR)(
            depth.unsqueeze(0)).squeeze(0)
        return image_resized, depth_resized

    def filter_depth(self, depth, ray_dir, ray_3d, uv):
        """过滤无效深度"""
        if not self.preprocess_config["depth_filter"]:
            return depth, ray_dir, ray_3d, uv

        min_depth = self.preprocess_config["min_depth"]
        max_depth = self.preprocess_config["max_depth"]
        valid_mask = (depth >= min_depth) & (depth <= max_depth)

        depth_valid = depth[valid_mask]
        ray_dir_valid = ray_dir[valid_mask]
        ray_3d_valid = ray_3d[valid_mask]
        uv_valid = (uv[0][valid_mask], uv[1][valid_mask])

        # 确保射线数不小于NUM_RAYS的80%
        if len(depth_valid) < BaseConfig.NUM_RAYS * 0.8:
            # 填充无效数据（用均值替代）
            fill_num = int(BaseConfig.NUM_RAYS - len(depth_valid))
            depth_fill = torch.full((fill_num,), depth_valid.mean(), device=depth_valid.device)
            ray_dir_fill = torch.zeros((fill_num, 3), device=ray_dir_valid.device)
            ray_3d_fill = torch.zeros((fill_num, 3), device=ray_3d_valid.device)
            uv_fill = (torch.zeros(fill_num, device=uv_valid[0].device),
                       torch.zeros(fill_num, device=uv_valid[1].device))

            depth_valid = torch.cat([depth_valid, depth_fill])
            ray_dir_valid = torch.cat([ray_dir_valid, ray_dir_fill])
            ray_3d_valid = torch.cat([ray_3d_valid, ray_3d_fill])
            uv_valid = (torch.cat([uv_valid[0], uv_fill[0]]),
                        torch.cat([uv_valid[1], uv_fill[1]]))

        return depth_valid[:BaseConfig.NUM_RAYS], ray_dir_valid[:BaseConfig.NUM_RAYS],
        ray_3d_valid[:BaseConfig.NUM_RAYS], (uv_valid[0][:BaseConfig.NUM_RAYS], uv_valid[1][:BaseConfig.NUM_RAYS])


def normalize_image(self, image):
    """图像归一化"""
    if self.preprocess_config["normalize"]:
        mean = torch.tensor(DataConfig.AUGMENT_CONFIG["mean"], device=image.device).view(3, 1, 1)
        std = torch.tensor(DataConfig.AUGMENT_CONFIG["std"], device=image.device).view(3, 1, 1)
        image = (image - mean) / std
    return image


def __call__(self, data):
    """完整预处理流程"""
    # 去畸变（简化）
    data["image"] = self.undistort_image(data["image"], data["K"], None)

    # 调整尺寸
    data["image"], data["depth"] = self.resize_data(data["image"], data["depth"])

    # 过滤深度
    data["depth"], data["ray_dir"], data["ray_3d"], data["uv"] = self.filter_depth(
        data["depth"], data["ray_dir"], data["ray_3d"], data["uv"]
    )

    # 归一化
    data["image"] = self.normalize_image(data["image"])

    return data


if __name__ == "__main__":
    # 测试预处理
    from data.real.data_parser import RealDataParserFactory
    from config.data_config import DataConfig

    parser = RealDataParserFactory.get_parser("euroc", DataConfig.REAL_DATASETS["euroc"])
    preprocessor = RealDataPreprocessor()

    img_names = sorted(os.listdir(parser.cam0_path))[:1]
    depth_names = sorted(os.listdir(parser.depth_path))[:1]
    data = parser.parse_data(img_names[0], depth_names[0])
    data_processed = preprocessor(data)

    print(f"预处理后图像形状: {data_processed['image'].shape}")
    print(f"预处理后射线数: {data_processed['ray_dir'].shape[0]}")
    print(f"预处理后深度范围: [{data_processed['depth'].min():.2f}, {data_processed['depth'].max():.2f}]")