"""Critic网络定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """Critic网络：输入全局观测和动作，输出Q值

    MADDPG采用中心化训练，Critic需要看到所有智能体的观测和动作

    Args:
        dim_info: 维度信息字典 {agent_id: [obs_dim, action_dim]}
        hidden_1: 第一层隐藏层维度，默认128
        hidden_2: 第二层隐藏层维度，默认128
    """

    def __init__(self, dim_info: dict, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        # 计算全局观测和动作维度
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a):
        """前向传播

        Args:
            s: 所有智能体的观测 (dict.values())
            a: 所有智能体的动作 (dict.values())

        Returns:
            Q值
        """
        # 传入全局观测和动作
        sa = torch.cat(list(s) + list(a), dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
