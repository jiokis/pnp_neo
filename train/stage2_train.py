import torch
from config.base_config import BaseConfig
from config.train_config import TrainConfig
from config.data_config import DataConfig
from data.dataloader.data_loader import build_data_loader
from train.trainer import Trainer

def main():
    """阶段二训练主脚本（尺度预测优化）"""
    # 打印配置信息
    print("="*60)
    print("                      无人机跨视角匹配 - 阶段二训练")
    print("="*60)
    BaseConfig.print_config()
    TrainConfig.print_config()
    DataConfig.print_config()

    # 检查GPU
    if BaseConfig.DEVICE.type != "cuda":
        print("警告：未检测到GPU，训练将非常缓慢！")
        input("按Enter继续，或Ctrl+C终止...")

    # 构建数据加载器
    print("\n构建数据加载器...")
    train_loader, train_sampler = build_data_loader(is_train=True)
    val_loader, _ = build_data_loader(is_train=False)  # 验证集使用训练集的10%
    print(f"训练加载器批次数量: {len(train_loader)}")
    print(f"验证加载器批次数量: {len(val_loader) if val_loader is not None else 0}")

    # 初始化训练器
    print("\n初始化训练器...")
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    # 设置随机种子，保证可复现性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 启动训练
    main()