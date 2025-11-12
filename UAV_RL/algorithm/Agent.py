"""Agent类：管理单个智能体的Actor和Critic网络"""
import torch
from copy import deepcopy
import sys
from pathlib import Path

# 添加父目录到路径以导入models
sys.path.append(str(Path(__file__).parent.parent))
from models import Actor, Critic


class Agent:
    """单个智能体的Actor-Critic管理器

    负责管理一个智能体的Actor网络、Critic网络及其目标网络、优化器

    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        dim_info: 所有智能体的维度信息（用于Critic）
        actor_lr: Actor学习率
        critic_lr: Critic学习率
        device: 训练设备
    """

    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, device):
        # 创建网络
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(dim_info).to(device)

        # 创建优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 创建目标网络
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        """更新Actor网络"""
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        """更新Critic网络"""
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
