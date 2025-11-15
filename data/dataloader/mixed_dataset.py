import os
import torch
import numpy as np
from torch.utils.data import Dataset
from config.data_config import DataConfig
from data.synthetic.data_generator import SyntheticDataGenerator
from data.synthetic.augment import SyntheticDataAugmentor


class MixedDroneDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.synth_generator = SyntheticDataGenerator()
        self.synth_augmentor = SyntheticDataAugmentor()
        self.synth_num = DataConfig.SYNTHETIC_NUM_PER_EPOCH
        self.total_num = self.synth_num
        if self.total_num <= 0:
            raise ValueError("合成数据量不能为0")

    def _force_cpu(self, data):
        """递归强制所有张量到CPU（生成后立即调用，根源拦截）"""
        if isinstance(data, torch.Tensor):
            return data.cpu()  # 不管什么设备，直接移到CPU
        elif isinstance(data, dict):
            return {k: self._force_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._force_cpu(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._force_cpu(v) for v in data)
        else:
            return data

    def _convert_to_tensor(self, data_dict):
        """只处理CPU上的数据，避免GPU冲突"""
        for k, v in data_dict.items():
            # 此时v已确保在CPU
            if not isinstance(v, torch.Tensor):
                try:
                    data_dict[k] = torch.tensor(v, dtype=torch.float32)
                except:
                    # 所有非张量都转numpy再转张量（CPU上操作）
                    data_dict[k] = torch.from_numpy(np.asarray(v)).float()
            else:
                data_dict[k] = v.float()
        return data_dict

    def _get_synthetic_data(self, index):
        # 1. 生成数据
        data_list, K = self.synth_generator.generate_data(num_views=2)

        # 2. 根源拦截：强制所有数据到CPU（关键！）
        data_list = self._force_cpu(data_list)
        K = self._force_cpu(K)

        # 3. 增强（此时数据已在CPU）
        processed_data = []
        for data in data_list:
            if self.is_train:
                data = self.synth_augmentor(data)
            # 4. 转换为CPU张量
            data = self._convert_to_tensor(data)
            processed_data.append(data)

        # 处理K（确保是CPU张量）
        if not isinstance(K, torch.Tensor):
            K = torch.from_numpy(np.asarray(K)).float()
        return processed_data, K

    def __getitem__(self, index):
        data_list, K = self._get_synthetic_data(index)
        batch_data = {
            "view0": data_list[0],  # 全程CPU
            "view1": data_list[1],  # 全程CPU
            "K": K  # 全程CPU
        }
        return batch_data

    def __len__(self):
        return self.total_num