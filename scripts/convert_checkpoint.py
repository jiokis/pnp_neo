import torch
from model.backbone.appearance_branch import MobileNetV3AppearanceBranch
from config.base_config import BaseConfig
import os


def convert_old_checkpoint(old_ckpt_path, new_ckpt_save_path):
    # 1. 初始化带特征金字塔的新模型
    new_model = MobileNetV3AppearanceBranch(output_dim=96).to(BaseConfig.DEVICE)

    # 2. 重新冻结backbone前3层
    for i, (name, param) in enumerate(new_model.backbone.named_parameters()):
        if i < 3:
            param.requires_grad = False
        else:
            break

    # 3. 加载旧checkpoint并过滤参数
    if not os.path.exists(old_ckpt_path):
        raise FileNotFoundError(f"旧模型权重不存在：{old_ckpt_path}")

    old_checkpoint = torch.load(old_ckpt_path, map_location=BaseConfig.DEVICE)
    old_state_dict = old_checkpoint["model_state_dict"]
    new_state_dict = new_model.state_dict()

    # 仅保留新旧模型共有的参数（跳过pyramid等新增模块）
    filtered_state_dict = {}
    for k, v in old_state_dict.items():
        if k in new_state_dict and new_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
        # else: 新增参数不加载，使用初始化值

    # 4. 更新并保存新模型权重
    new_state_dict.update(filtered_state_dict)
    new_model.load_state_dict(new_state_dict)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(new_ckpt_save_path), exist_ok=True)
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "epoch": old_checkpoint.get("epoch", 0),  # 继承epoch数
        "optimizer_state_dict": old_checkpoint.get("optimizer_state_dict")  # 可选保留优化器
    }, new_ckpt_save_path)

    print(f"✅ 转换完成！新模型权重已保存至：{new_ckpt_save_path}")
    print(f"保留旧参数数量：{len(filtered_state_dict)}/{len(old_state_dict)}")


if __name__ == "__main__":
    # 替换为你的实际路径
    OLD_CHECKPOINT_PATH = "model/checkpoints/old_model_epoch10.pth"  # 旧模型权重路径
    NEW_CHECKPOINT_PATH = "model/checkpoints/main_model_with_pyramid_init.pth"  # 新模型保存路径
    convert_old_checkpoint(OLD_CHECKPOINT_PATH, NEW_CHECKPOINT_PATH)