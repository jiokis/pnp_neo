import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config.base_config import BaseConfig


class LoRACameraAdaptor(nn.Module):
    def __init__(self, in_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, in_dim))
        nn.init.zeros_(self.B)  # 初始不影响原始特征

    def forward(self, x):
        # x: (B×V, 96)
        lora_delta = x @ self.A @ self.B  # (B×V, 96)
        return x + lora_delta  # 残差微调