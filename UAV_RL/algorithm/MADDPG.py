"""MADDPG算法实现

论文：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
链接：https://arxiv.org/abs/1706.02275
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from algorithm.Agent import Agent
from utils.Buffer import Buffer


class MADDPG:
    """MADDPG算法类

    多智能体深度确定性策略梯度算法
    - 中心化训练：Critic看到所有智能体的观测和动作
    - 分布式执行：Actor只使用自己的观测

    Args:
        dim_info: 维度信息字典 {agent_id: [obs_dim, action_dim]}
        is_continue: 是否为连续动作空间
        actor_lr: Actor学习率
        critic_lr: Critic学习率
        buffer_size: 经验回放缓冲区大小
        device: 训练设备
    """

    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device):
        self.agents = {}
        self.buffers = {}

        # 为每个智能体创建Agent和Buffer
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(
                obs_dim, action_dim, dim_info,
                actor_lr, critic_lr, device=device
            )
            self.buffers[agent_id] = Buffer(
                buffer_size, obs_dim,
                act_dim=action_dim if is_continue else 1,
                device=device
            )

        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0]  # sample时使用

    def select_action(self, obs):
        """选择动作

        Args:
            obs: 观测字典 {agent_id: obs}

        Returns:
            动作字典 {agent_id: action}
        """
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            action = self.agents[agent_id].actor(obs)
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
        return actions

    def add(self, obs, action, reward, next_obs, done):
        """添加经验到回放缓冲区

        Args:
            obs: 观测字典
            action: 动作字典
            reward: 奖励字典
            next_obs: 下一观测字典
            done: 结束标志字典
        """
        for agent_id, buffer in self.buffers.items():
            buffer.add(
                obs[agent_id],
                action[agent_id],
                reward[agent_id],
                next_obs[agent_id],
                done[agent_id]
            )

    def sample(self, batch_size):
        """从缓冲区采样

        Args:
            batch_size: 批量大小

        Returns:
            (obs, action, reward, next_obs, done, next_action)
        """
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        next_action = {}

        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)
            next_action[agent_id] = self.agents[agent_id].actor_target(next_obs[agent_id])

        return obs, action, reward, next_obs, done, next_action

    def learn(self, batch_size, gamma, tau):
        """学习更新

        Args:
            batch_size: 批量大小
            gamma: 折扣因子
            tau: 软更新系数
        """
        # 多智能体特有 - 集中式训练critic
        for agent_id, agent in self.agents.items():
            obs, action, reward, next_obs, done, next_action = self.sample(batch_size)
            next_target_Q = agent.critic_target(next_obs.values(), next_action.values())

            # 更新Critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            # 更新Actor
            new_action = agent.actor(obs[agent_id])
            action[agent_id] = new_action
            actor_loss = -agent.critic(obs.values(), action.values()).mean()
            agent.update_actor(actor_loss)

        self.update_target(tau)

    def update_target(self, tau):
        """软更新目标网络

        Args:
            tau: 软更新系数
        """
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_dir):
        """保存模型

        Args:
            model_dir: 保存目录
        """
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_dir, 'MADDPG.pth')
        )

    @staticmethod
    def load(dim_info, is_continue, model_dir):
        """加载模型

        Args:
            dim_info: 维度信息
            is_continue: 是否连续动作
            model_dir: 模型目录

        Returns:
            加载的MADDPG策略
        """
        policy = MADDPG(
            dim_info, is_continue=is_continue,
            actor_lr=0, critic_lr=0, buffer_size=0, device='cpu'
        )
        data = torch.load(os.path.join(model_dir, 'MADDPG.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return policy
