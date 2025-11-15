import torch
from config.base_config import BaseConfig


class Real2SyntheticConverter:
    """真实数据格式转换为合成数据格式"""

    def __init__(self):
        self.num_rays = BaseConfig.NUM_RAYS

    def convert(self, real_data):
        """转换单视角数据"""
        # 确保射线数一致
        if real_data["ray_dir"].shape[0] < self.num_rays:
            # 填充数据
            fill_num = self.num_rays - real_data["ray_dir"].shape[0]
            ray_dir_fill = torch.zeros((fill_num, 3), device=real_data["ray_dir"].device)
            ray_3d_fill = torch.zeros((fill_num, 3), device=real_data["ray_3d"].device)
            depth_fill = torch.full((fill_num,), 15.0, device=real_data["depth"].device)  # 默认深度15m
            uv_fill = (torch.zeros(fill_num, device=real_data["uv"][0].device),
                       torch.zeros(fill_num, device=real_data["uv"][1].device))

            real_data["ray_dir"] = torch.cat([real_data["ray_dir"], ray_dir_fill])
            real_data["ray_3d"] = torch.cat([real_data["ray_3d"], ray_3d_fill])
            real_data["depth"] = torch.cat([real_data["depth"], depth_fill])
            real_data["uv"] = (torch.cat([real_data["uv"][0], uv_fill[0]]),
                               torch.cat([real_data["uv"][1], uv_fill[1]]))
        elif real_data["ray_dir"].shape[0] > self.num_rays:
            # 裁剪数据
            real_data["ray_dir"] = real_data["ray_dir"][:self.num_rays]
            real_data["ray_3d"] = real_data["ray_3d"][:self.num_rays]
            real_data["depth"] = real_data["depth"][:self.num_rays]
            real_data["uv"] = (real_data["uv"][0][:self.num_rays],
                               real_data["uv"][1][:self.num_rays])

        # 统一输出格式（与合成数据一致）
        synthetic_format_data = {
            "image": real_data["image"],
            "ray_dir": real_data["ray_dir"],
            "ray_3d": real_data["ray_3d"],
            "depth": real_data["depth"],
            "K": real_data["K"],
            "rot_mat": real_data["rot_mat"],
            "uv": real_data["uv"]
        }

        return synthetic_format_data

    def batch_convert(self, real_data_batch):
        """批量转换"""
        synthetic_batch = []
        for views_data in real_data_batch:
            converted_views = [self.convert(view_data) for view_data in views_data]
            synthetic_batch.append(converted_views)
        return synthetic_batch


if __name__ == "__main__":
    # 测试转换
    from data.real.data_parser import RealDataParserFactory
    from data.real.preprocess import RealDataPreprocessor
    from config.data_config import DataConfig

    parser = RealDataParserFactory.get_parser("euroc", DataConfig.REAL_DATASETS["euroc"])
    preprocessor = RealDataPreprocessor()
    converter = Real2SyntheticConverter()

    img_names = sorted(os.listdir(parser.cam0_path))[:2]
    depth_names = sorted(os.listdir(parser.depth_path))[:2]
    real_views = [preprocessor(parser.parse_data(img, depth)) for img, depth in zip(img_names, depth_names)]
    converted_views = converter.batch_convert([real_views])[0]

    print(f"转换后视角数: {len(converted_views)}")
    print(f"转换后射线数: {converted_views[0]['ray_dir'].shape[0]}")
    print(f"格式统一验证: {list(converted_views[0].keys())}")