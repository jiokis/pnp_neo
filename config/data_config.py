# 数据参数配置
class DataConfig:
    # 数据比例（合成:真实=7:3）
    SYNTHETIC_RATIO = 1.0
    REAL_RATIO = 0.0

    # 合成数据配置
    SYNTHETIC_DATA_PATH = "./data/synthetic/generated_data/"
    SYNTHETIC_NUM_PER_EPOCH = 400  # 每轮合成数据量（70%×400）

    # 真实数据配置
    REAL_DATASETS = {
        "euroc": "./data/real/EuRoC/",  # EuRoC数据集路径
        "tum": "./data/real/TUM/"       # TUM数据集路径
    }
    REAL_NUM_PER_EPOCH = 0  # 每轮真实数据量（30%×400）

    # 数据增强配置
    AUGMENT_CONFIG = {
        "random_rotate": True,
        "rotate_range": (-10, 10),  # 随机旋转范围（度）
        "brightness_jitter": True,
        "brightness_range": (0.8, 1.2),  # 亮度调整范围
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],  # RGB归一化均值
        "std": [0.229, 0.224, 0.225]    # RGB归一化标准差
    }

    # 预处理配置
    PREPROCESS_CONFIG = {
        "undistort": True,  # 图像去畸变
        "resize": (640, 480),  # 统一尺寸
        "depth_filter": True,  # 深度过滤
        "min_depth": 3.0,
        "max_depth": 30.0
    }

    @staticmethod
    def print_config():
        print("="*50)
        print("数据配置参数")
        print("="*50)
        print(f"数据比例: 合成{DataConfig.SYNTHETIC_RATIO} | 真实{DataConfig.REAL_RATIO}")
        print(f"合成数据路径: {DataConfig.SYNTHETIC_DATA_PATH}")
        print(f"真实数据集: {list(DataConfig.REAL_DATASETS.keys())}")
        print(f"数据增强: {DataConfig.AUGMENT_CONFIG}")
        print("="*50)