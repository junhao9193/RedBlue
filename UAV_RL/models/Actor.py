"""Actor网络定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor网络：输入观测，输出动作

    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        hidden_1: 第一层隐藏层维度，默认128
        hidden_2: 第二层隐藏层维度，默认128
    """

    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))  # 输出[-1, 1]
        return x
