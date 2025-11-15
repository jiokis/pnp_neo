import torch

class GPUUtils:
    """GPU显存优化工具"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_gpu_memory_usage(self, prefix=""):
        """打印GPU显存使用情况"""
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3     # GB
        print(f"{prefix} GPU显存使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

    def auto_adjust_batch_size(self, model, initial_batch_size, input_data):
        """自动调整批次大小以适应显存"""
        if self.device.type != "cuda":
            return initial_batch_size

        batch_size = initial_batch_size
        while batch_size > 0:
            try:
                # 测试批次大小
                test_input = {k: v[:batch_size].to(self.device) for k, v in input_data.items()}
                model(test_input)
                print(f"自动适配批次大小: {batch_size}")
                return batch_size
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    print(f"显存不足，批次大小调整为: {batch_size}")
                    if batch_size == 0:
                        raise MemoryError("最小批次大小仍无法适配显存")
                else:
                    raise e
        return 1

    def clear_gpu_cache(self):
        """清理GPU缓存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("GPU缓存已清理")