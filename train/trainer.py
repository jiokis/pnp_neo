import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import csv
import glob  # 新增：用于查找Stage2目录权重文件
from tqdm import tqdm
from config.base_config import BaseConfig
from config.train_config import TrainConfig
from model.full_model import DroneCrossViewModel
from train.loss_functions import LossFunctions
from utils.eval_utils import EvaluationMetrics
from utils.log_utils import Logger
from utils.gpu_utils import GPUUtils
# train/trainer.py 开头导入部分
from utils.pyramid_monitor import PyramidMonitor  # 新增这行


class Trainer:
    """训练器类（阶段二专用，集成特征金字塔支持）"""

    def __init__(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = BaseConfig.DEVICE
        self.train_config = TrainConfig
        self.loss_weights = TrainConfig.LOSS_WEIGHTS

        # 初始化模型
        self.model = DroneCrossViewModel().to(self.device)
        # 初始化损失函数
        self.loss_fn = LossFunctions()
        # 初始化优化器
        self.optimizer = self._build_optimizer()
        # 初始化学习率调度器
        self.scheduler = self._build_scheduler()
        # 初始化混合精度缩放器
        self.scaler = GradScaler() if BaseConfig.DTYPE == torch.float16 else None
        # 初始化评估指标
        self.eval_metrics = EvaluationMetrics()
        # 初始化日志记录
        self.logger = Logger(TrainConfig.TRAIN_LOG_PATH, TrainConfig.TENSORBOARD_LOG_PATH)
        # 初始化GPU工具
        self.gpu_utils = GPUUtils()

        # 创建权重保存目录
        os.makedirs(TrainConfig.STAGE2_SAVE_PATH, exist_ok=True)

        # ---------------------- 关键修改：加载Stage2带特征金字塔的权重 ----------------------
        self.start_epoch = self._load_stage2_pyramid_weights()
        # ----------------------------------------------------------------------------------
        self.pyramid_monitor = PyramidMonitor(self.model, log_path=TrainConfig.TRAIN_LOG_PATH)

    def _build_optimizer(self):
        """构建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.LR,
            weight_decay=1e-4
        )

    def _build_scheduler(self):
        """构建学习率调度器"""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.train_config.LR_DECAY_STEP,
            gamma=self.train_config.LR_DECAY_GAMMA,
        )

    def _load_stage1_weights(self):
        """加载阶段一预训练权重（为空则跳过，避免不匹配错误）- 保留原方法，避免影响其他逻辑"""
        if self.train_config.STAGE1_WEIGHT_PATH and os.path.exists(self.train_config.STAGE1_WEIGHT_PATH):
            checkpoint = torch.load(self.train_config.STAGE1_WEIGHT_PATH, map_location=self.device)
            model_dict = self.model.state_dict()
            # 只加载形状匹配的权重
            pretrained_dict = {k: v for k, v in checkpoint.items() if
                               k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)  # strict=False允许跳过不匹配层
            print(f"成功加载匹配的预训练权重（共加载 {len(pretrained_dict)}/{len(model_dict)} 个参数）")
        else:
            print("未使用预训练权重，模型参数随机初始化")

    # ---------------------- 新增：加载Stage2带特征金字塔的权重方法 ----------------------
    def _load_stage2_pyramid_weights(self):
        """加载Stage2目录中带特征金字塔的权重（优先加载，适配当前训练）"""
        # 查找Stage2目录中所有pth权重文件
        checkpoint_files = glob.glob(os.path.join(self.train_config.STAGE2_SAVE_PATH, "*.pth"))
        target_ckpt_path = None

        # 优先选择转换后的带金字塔初始化权重（含"pyramid_init"标识）
        for ckpt in checkpoint_files:
            if "main_model_with_pyramid_init" in ckpt:
                target_ckpt_path = ckpt
                break

        # 若没有专门的初始化权重，选择最新的epoch权重（按文件名排序）
        if not target_ckpt_path and checkpoint_files:
            # 按epoch数排序，取最后一个（最新的）
            checkpoint_files.sort(
                key=lambda x: int(os.path.basename(x).split("_epoch_")[-1].split("_")[0])
                if "epoch_" in x else 0
            )
            target_ckpt_path = checkpoint_files[-1]

        if target_ckpt_path and os.path.exists(target_ckpt_path):
            checkpoint = torch.load(target_ckpt_path, map_location=self.device)
            # 兼容完整checkpoint格式（含model_state_dict键）或直接是state_dict
            pretrained_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

            model_dict = self.model.state_dict()
            # 只加载「键存在且形状匹配」的参数（自动跳过金字塔等新增模块参数）
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict, strict=False)

            # 强制冻结backbone前3层（保护原有特征提取权重，避免被破坏）
            for i, (name, param) in enumerate(self.model.backbone.appearance_branch.backbone.named_parameters()):
                if i < 3:
                    param.requires_grad = False
                else:
                    break

            print(f"✅ 成功加载Stage2带特征金字塔的权重！")
            print(f"加载路径: {target_ckpt_path}")
            print(f"加载参数数量：{len(filtered_dict)}/{len(model_dict)}（新增模块用初始化值）")
            # 返回上次训练的epoch，若无则返回0
            return checkpoint.get("epoch", 0)
        else:
            print("⚠️ Stage2目录中无带特征金字塔的权重，模型参数随机初始化")
            # 无权重时也冻结backbone前3层
            for i, (name, param) in enumerate(self.model.backbone.appearance_branch.backbone.named_parameters()):
                if i < 3:
                    param.requires_grad = False
                else:
                    break
            return 0  # 从epoch 0开始训练

    # ----------------------------------------------------------------------------------

    def _train_one_batch(self, batch_data):
        """训练单批次数据"""
        view0_data = {k: v.to(self.device) for k, v in batch_data["view0"].items()}
        view1_data = {k: v.to(self.device) for k, v in batch_data["view1"].items()}
        K = batch_data["K"].to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        # 整理输入数据（合并多视角，适配模型输入格式）
        # view0数据
        view0_data = {
            "ray_dir": batch_data["view0"]["ray_dir"].to(self.device),
            "ray_3d": batch_data["view0"]["ray_3d"].to(self.device),
            "image": batch_data["view0"]["image"].to(self.device),
            "depth": batch_data["view0"]["depth"].to(self.device)
        }
        # view1数据（仅用于后续跨视角评估，训练阶段用view0）
        view1_data = {
            "ray_dir": batch_data["view1"]["ray_dir"].to(self.device),
            "ray_3d": batch_data["view1"]["ray_3d"].to(self.device),
            "image": batch_data["view1"]["image"].to(self.device),
            "depth": batch_data["view1"]["depth"].to(self.device)
        }

        # 混合精度训练
        if self.scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_dict = self.model(view0_data)
                total_loss, loss_dict = self.loss_fn.compute_total_loss(
                    pred_dict, view0_data, self.loss_weights
                )
            # 梯度缩放与反向传播
            self.scaler.scale(total_loss).backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.MAX_GRAD_NORM)
            # 优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred_dict = self.model(view0_data)
            total_loss, loss_dict = self.loss_fn.compute_total_loss(
                pred_dict, view0_data, self.loss_weights
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.MAX_GRAD_NORM)
            self.optimizer.step()

        # 计算评估指标
        metrics = self.eval_metrics.compute_metrics(
            pred_dict=pred_dict,
            gt_data=view0_data,
            view1_data=view1_data,
            pred_dict_view1=self.model(view1_data) if self.val_loader else None
        )

        return loss_dict, metrics

    def _validate_one_epoch(self):
        """验证单轮次"""
        if self.val_loader is None:
            return None
        self.model.eval()
        val_loss_dict = {}
        val_metrics = {}
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # 整理数据
                view0_data = {
                    "ray_dir": batch_data["view0"]["ray_dir"].to(self.device),
                    "ray_3d": batch_data["view0"]["ray_3d"].to(self.device),
                    "image": batch_data["view0"]["image"].to(self.device),
                    "depth": batch_data["view0"]["depth"].to(self.device)
                }
                view1_data = {
                    "ray_dir": batch_data["view1"]["ray_dir"].to(self.device),
                    "ray_3d": batch_data["view1"]["ray_3d"].to(self.device),
                    "image": batch_data["view1"]["image"].to(self.device),
                    "depth": batch_data["view1"]["depth"].to(self.device)
                }

                # 预测
                pred_dict = self.model(view0_data)
                pred_dict_view1 = self.model(view1_data)

                # 计算损失
                _, loss_dict = self.loss_fn.compute_total_loss(
                    pred_dict, view0_data, self.loss_weights
                )

                # 计算指标
                metrics = self.eval_metrics.compute_metrics(
                    pred_dict=pred_dict,
                    gt_data=view0_data,
                    view1_data=view1_data,
                    pred_dict_view1=pred_dict_view1
                )

                # 累积损失和指标
                for k, v in loss_dict.items():
                    val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v

        # 计算平均值
        batch_num = len(self.val_loader)
        val_loss_dict = {k: v / batch_num for k, v in val_loss_dict.items()}
        val_metrics = {k: v / batch_num for k, v in val_metrics.items()}

        return val_loss_dict, val_metrics

    def train(self):
        """开始训练"""
        print("=" * 50)
        print("开始阶段二训练（尺度预测优化+特征金字塔）")
        print("=" * 50)
        BaseConfig.print_config()
        TrainConfig.print_config()

        best_val_loss = float('inf')
        # ---------------------- 关键修改：从上次训练的epoch+1开始 ----------------------
        for epoch in range(self.start_epoch + 1, self.train_config.EPOCHS + 1):
            # ----------------------------------------------------------------------------------
            print(f"\nEpoch [{epoch}/{self.train_config.EPOCHS}]")
            epoch_loss_dict = {}
            epoch_metrics = {}

            # 训练单轮
            train_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
            for batch_idx, batch_data in enumerate(train_bar):
                # 梯度累积
                loss_dict, metrics = self._train_one_batch(batch_data)

                # 累积损失和指标
                for k, v in loss_dict.items():
                    epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + v.item()
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

                # 打印批次信息
                train_bar.set_postfix({
                    "total_loss": round(loss_dict["total_loss"].item(), 4),
                    "reg_loss": round(loss_dict["reg_loss"].item(), 4),
                    "lr": round(self.optimizer.param_groups[0]["lr"], 6)
                })

                # 显存监控
                if (batch_idx + 1) % 10 == 0:
                    self.gpu_utils.print_gpu_memory_usage(f"Batch {batch_idx + 1}")

            # 计算训练集平均值
            batch_num = len(self.train_loader)
            epoch_loss_dict = {k: v / batch_num for k, v in epoch_loss_dict.items()}
            epoch_metrics = {k: v / batch_num for k, v in epoch_metrics.items()}

            # 验证
            val_loss_dict, val_metrics = self._validate_one_epoch()

            # 学习率调度
            self.scheduler.step()

            # 日志记录
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=epoch_loss_dict,
                train_metrics=epoch_metrics,
                val_loss=val_loss_dict,
                val_metrics=val_metrics
            )

            # 打印 epoch 总结
            print(f"\nEpoch {epoch} 总结:")
            print("训练损失:", end=" ")
            for k, v in epoch_loss_dict.items():
                print(f"{k}: {v:.4f}", end=" ")
            print("\n训练指标:", end=" ")
            for k, v in epoch_metrics.items():
                print(f"{k}: {v:.4f}", end=" ")
            if val_loss_dict is not None:
                print("\n验证损失:", end=" ")
                for k, v in val_loss_dict.items():
                    print(f"{k}: {v:.4f}", end=" ")
                print("\n验证指标:", end=" ")
                for k, v in val_metrics.items():
                    print(f"{k}: {v:.4f}", end=" ")

            # 新增：特征金字塔监控（记录参数更新和性能增益）
            self.pyramid_monitor.monitor_epoch(epoch, epoch_loss_dict, epoch_metrics, val_loss_dict, val_metrics)

            # 保存权重
            if epoch % self.train_config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    self.train_config.STAGE2_SAVE_PATH,
                    f"epoch_{epoch}_checkpoint.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": epoch_loss_dict,
                    "val_loss": val_loss_dict
                }, checkpoint_path)
                print(f"\n已保存权重: {checkpoint_path}")

            # 保存最优模型
            if val_loss_dict is not None and val_loss_dict["total_loss"] < best_val_loss:
                best_val_loss = val_loss_dict["total_loss"]
                best_checkpoint_path = os.path.join(
                    self.train_config.STAGE2_SAVE_PATH,
                    "best_val_checkpoint.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics
                }, best_checkpoint_path)
                print(f"更新最优模型: {best_checkpoint_path}")

        # 训练结束
        print("\n" + "=" * 50)
        print("阶段二训练完成！")
        print("=" * 50)
        self.logger.close()