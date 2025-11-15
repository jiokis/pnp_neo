import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from config.train_config import TrainConfig
from data.dataloader.mixed_dataset import MixedDroneDataset


def build_data_loader(is_train=True):
    """构建数据加载器"""
    dataset = MixedDroneDataset(is_train=is_train)

    # 采样器配置
    if is_train and torch.cuda.device_count() > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    # 数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=TrainConfig.BATCH_SIZE if is_train else TrainConfig.VAL_BATCH_SIZE,
        shuffle=(sampler is None) and is_train,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    return data_loader, sampler


if __name__ == "__main__":
    # 测试数据加载器
    train_loader, _ = build_data_loader(is_train=True)
    print(f"训练加载器批次数量: {len(train_loader)}")

    # 迭代测试
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  View0图像形状: {batch_data['view0']['image'].shape}")
        print(f"  View0射线方向形状: {batch_data['view0']['ray_dir'].shape}")
        print(f"  View1深度形状: {batch_data['view1']['depth'].shape}")
        if batch_idx >= 1:
            break