import torch

# 基础参数配置
class BaseConfig:
    # 图像参数
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    CHANNELS = 3

    # 射线参数
    NUM_RAYS = 512  # 适配8GB GPU
    MIN_DEPTH = 3.0  # 最小有效深度
    MAX_DEPTH = 30.0  # 最大有效深度

    # 锚框参数
    ANCHOR_CENTERS = torch.tensor([4.5, 9.0, 18.0, 27.0])  # 4个锚框中心
    ANCHOR_NUM = 4

    # 模型参数
    LATENT_DIM = 128  # 隐向量维度
    TRANSFORMER_HEADS = 1  # 注意力头数（适配显存）
    LOCAL_RADIUS = 0.6  # 局部注意力半径

    # 设备参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    @staticmethod
    def print_config():
        print("="*50)
        print("基础配置参数")
        print("="*50)
        print(f"图像尺寸: {BaseConfig.IMG_WIDTH}×{BaseConfig.IMG_HEIGHT}")
        print(f"射线数: {BaseConfig.NUM_RAYS}")
        print(f"深度范围: [{BaseConfig.MIN_DEPTH}, {BaseConfig.MAX_DEPTH}]")
        print(f"设备: {BaseConfig.DEVICE}")
        print(f"数据类型: {BaseConfig.DTYPE}")
        print("="*50)