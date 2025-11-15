import torch
import csv
import os
from datetime import datetime


class PyramidMonitor:
    """特征金字塔训练监控器：实时跟踪参数更新、性能增益，生成监控日志"""

    def __init__(self, model, log_path="./logs/pyramid_monitor.csv"):
        self.model = model
        self.log_path = log_path
        self.pyramid_params = self._get_pyramid_params()  # 记录金字塔模块参数名
        self.init_pyramid_weights = self._save_init_weights()  # 保存初始权重（用于计算更新量）
        self._init_log_file()  # 初始化监控日志文件

    def _get_pyramid_params(self):
        """获取特征金字塔模块的所有参数名（过滤其他模块）"""
        pyramid_param_names = []
        for name, _ in self.model.named_parameters():
            # 匹配特征金字塔模块的参数（对应appearance_branch中的pyramid层）
            if "backbone.appearance_branch.pyramid" in name:
                pyramid_param_names.append(name)
        return pyramid_param_names

    def _save_init_weights(self):
        """保存特征金字塔模块的初始权重（用于后续计算参数变化）"""
        init_weights = {}
        for name in self.pyramid_params:
            param = dict(self.model.named_parameters())[name]
            init_weights[name] = param.data.clone().detach()
        return init_weights

    def _init_log_file(self):
        """初始化监控日志文件，写入表头"""
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 若日志文件不存在，写入表头
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "datetime", "epoch",
                    "pyramid_param_update_ratio",  # 金字塔参数更新比例
                    "train_loss_change",  # 训练损失变化（相对上一轮）
                    "val_loss_change",  # 验证损失变化（相对上一轮）
                    "key_metric_change"  # 关键指标变化（如AUC/准确率）
                ])

    def _calc_param_update_ratio(self):
        """计算特征金字塔参数的更新比例（L2范数相对初始权重的变化）"""
        if not self.pyramid_params:
            return 0.0

        total_update = 0.0
        total_init = 0.0
        for name in self.pyramid_params:
            current_param = dict(self.model.named_parameters())[name].data
            init_param = self.init_pyramid_weights[name]

            # 计算L2范数
            update_norm = torch.norm(current_param - init_param).item()
            init_norm = torch.norm(init_param).item()

            total_update += update_norm
            total_init += init_norm

        # 避免除以0（初始权重全0时）
        return total_update / total_init if total_init > 1e-8 else 0.0

    def monitor_epoch(self, epoch, train_loss, train_metrics, val_loss=None, val_metrics=None):
        """每轮epoch结束后，记录监控数据"""
        # 1. 计算金字塔参数更新比例
        update_ratio = self._calc_param_update_ratio()

        # 2. 计算损失/指标变化（相对上一轮，首次epoch记为0）
        if epoch == 1:
            train_loss_change = 0.0
            val_loss_change = 0.0
            key_metric_change = 0.0
        else:
            # 读取上一轮日志数据
            with open(self.log_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                last_row = list(reader)[-1]  # 最后一行是上一轮数据

            # 训练损失变化（当前 - 上一轮）
            train_loss_change = train_loss["total_loss"] - float(last_row[2])
            # 验证损失变化
            val_loss_change = val_loss["total_loss"] - float(last_row[3]) if val_loss else 0.0
            # 关键指标变化（假设用第一个指标，如准确率/auc，当前 - 上一轮）
            key_metric = list(train_metrics.values())[0] if train_metrics else 0.0
            key_metric_change = key_metric - float(last_row[4])

        # 3. 写入日志
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                round(update_ratio, 6),
                round(train_loss_change, 4),
                round(val_loss_change, 4) if val_loss else 0.0,
                round(key_metric_change, 4)
            ])

        # 4. 打印实时监控信息（终端输出，方便查看）
        print(f"\n📊 特征金字塔监控（Epoch {epoch}）:")
        print(f"   - 参数更新比例: {update_ratio:.6f}（>0表示模块正在训练）")
        print(f"   - 训练损失变化: {train_loss_change:.4f}（负值表示下降）")
        if val_loss:
            print(f"   - 验证损失变化: {val_loss_change:.4f}（负值表示下降）")
        print(f"   - 关键指标变化: {key_metric_change:.4f}（正值表示提升）")

    def get_summary(self):
        """获取训练至今的监控总结"""
        with open(self.log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]  # 跳过表头

        total_epochs = len(rows)
        final_update_ratio = float(rows[-1][1]) if total_epochs > 0 else 0.0
        final_train_loss = float(rows[-1][2]) if total_epochs > 0 else 0.0
        final_val_loss = float(rows[-1][3]) if total_epochs > 0 else 0.0

        summary = f"""
        📈 特征金字塔训练监控总结（共{total_epochs}轮）:
        - 最终参数更新比例: {final_update_ratio:.6f}
        - 最终训练损失: {final_train_loss:.4f}
        - 最终验证损失: {final_val_loss:.4f}
        - 模块训练有效性: {'✅ 有效（参数更新正常）' if final_update_ratio > 1e-6 else '❌ 无效（参数未更新）'}
        """
        print(summary)
        return summary