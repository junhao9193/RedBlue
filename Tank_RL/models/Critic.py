import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Critic 网络：集中式 Q 网络
    输入所有智能体的观测和动作，输出 Q 值
    """
    def __init__(self, dim_info: dict, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        # 计算全局观测和动作的维度
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a):
        """
        Args:
            s: 所有智能体的观测
            a: 所有智能体的动作
        Returns:
            q: Q 值
        """
        sa = torch.cat(list(s) + list(a), dim=1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q
