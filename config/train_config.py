from config.base_config import BaseConfig

# 训练参数配置
class TrainConfig:
    # 批次参数
    BATCH_SIZE = 4  # 适配8GB GPU
    VAL_BATCH_SIZE = 2
    NUM_WORKERS = 4  # 数据加载线程数

    # 学习率参数
    LR = 3e-5  # 初始学习率
    LR_DECAY_STEP = 10  # 学习率衰减步长
    LR_DECAY_GAMMA = 0.5  # 学习率衰减系数

    # 训练轮次
    EPOCHS = 40  # 阶段二总轮次
    LOG_INTERVAL = 1  # 日志打印间隔
    SAVE_INTERVAL = 5  # 权重保存间隔

    # 损失权重（阶段二：强化尺度预测）
    LOSS_WEIGHTS = {
        "cls_loss": 0.2,
        "pos_loss": 0.3,
        "reg_loss": 0.4,
        "anchor_center_loss": 0.1
    }

    # 梯度优化
    GRAD_ACCUM_STEPS = 2  # 梯度累积步数（适配显存）
    MAX_GRAD_NORM = 1.0  # 梯度裁剪阈值

    # 权重路径
    STAGE1_WEIGHT_PATH = ""
    STAGE2_SAVE_PATH = "./weights/stage2_checkpoint/"

    # 日志路径
    TRAIN_LOG_PATH = "./logs/train_metrics.csv"
    TENSORBOARD_LOG_PATH = "./logs/tensorboard/"

    @staticmethod
    def print_config():
        print("="*50)
        print("训练配置参数")
        print("="*50)
        print(f"批次大小: {TrainConfig.BATCH_SIZE} (梯度累积×{TrainConfig.GRAD_ACCUM_STEPS})")
        print(f"初始学习率: {TrainConfig.LR}")
        print(f"训练轮次: {TrainConfig.EPOCHS}")
        print(f"损失权重: {TrainConfig.LOSS_WEIGHTS}")
        print(f"预训练权重: {TrainConfig.STAGE1_WEIGHT_PATH}")
        print("="*50)