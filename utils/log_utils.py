import os
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """日志记录工具"""
    def __init__(self, csv_path, tensorboard_path):
        self.csv_path = csv_path
        self.tensorboard_path = tensorboard_path

        # 创建TensorBoard writer
        os.makedirs(tensorboard_path, exist_ok=True)
        self.tb_writer = SummaryWriter(tensorboard_path)

        # 创建CSV文件
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = None
        self.field_names = None

    def _init_csv(self, loss_keys, metric_keys):
        """初始化CSV字段"""
        self.field_names = ["epoch", "lr"]
        # 训练损失
        for k in loss_keys:
            self.field_names.append(f"train_{k}")
        # 训练指标
        for k in metric_keys:
            self.field_names.append(f"train_{k}")
        # 验证损失和指标
        for k in loss_keys:
            self.field_names.append(f"val_{k}")
        for k in metric_keys:
            self.field_names.append(f"val_{k}")

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.field_names)
        self.csv_writer.writeheader()

    def log_epoch(self, epoch, train_loss, train_metrics, val_loss=None, val_metrics=None, lr=0.0):
        """记录单轮日志"""
        # 初始化CSV（首次调用）
        if self.csv_writer is None:
            loss_keys = list(train_loss.keys())
            metric_keys = list(train_metrics.keys())
            self._init_csv(loss_keys, metric_keys)

        # 构建日志字典
        log_dict = {
            "epoch": epoch,
            "lr": lr
        }

        # 添加训练损失和指标
        for k, v in train_loss.items():
            log_dict[f"train_{k}"] = v
            self.tb_writer.add_scalar(f"Train/Loss_{k}", v, epoch)
        for k, v in train_metrics.items():
            log_dict[f"train_{k}"] = v
            self.tb_writer.add_scalar(f"Train/Metric_{k}", v, epoch)

        # 添加验证损失和指标
        if val_loss is not None:
            for k, v in val_loss.items():
                log_dict[f"val_{k}"] = v
                self.tb_writer.add_scalar(f"Val/Loss_{k}", v, epoch)
        if val_metrics is not None:
            for k, v in val_metrics.items():
                log_dict[f"val_{k}"] = v
                self.tb_writer.add_scalar(f"Val/Metric_{k}", v, epoch)

        # 写入CSV
        self.csv_writer.writerow(log_dict)
        self.csv_file.flush()

        # TensorBoard刷新
        self.tb_writer.flush()

    def close(self):
        """关闭日志资源"""
        self.csv_file.close()
        self.tb_writer.close()
        print(f"日志已保存至: {self.csv_path} 和 {self.tensorboard_path}")