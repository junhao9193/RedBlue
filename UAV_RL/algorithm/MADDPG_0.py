"""
UAV电磁对抗环境的MADDPG训练脚本（简化版）
基于env_restruction.py的MADDPG_simple.py改编

论文：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
链接：https://arxiv.org/abs/1706.02275
"""
import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import sys
from pathlib import Path
# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from Buffer import Buffer

from copy import deepcopy
import gymnasium as gym
import argparse

## 其他
from torch.utils.tensorboard import SummaryWriter
import time
import re
import math

### 自定义环境
sys.path.append(str(Path(__file__).parent.parent))
from env_uav import UAVEnv


## 第一部分：定义Actor和Critic网络
class Actor(nn.Module):
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


class Critic(nn.Module):
    def __init__(self, dim_info: dict, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        # 计算全局观测和动作维度
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a):
        # 传入全局观测和动作
        sa = torch.cat(list(s) + list(a), dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class Agent:
    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, device):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(dim_info).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


## 第二部分：定义MADDPG算法类
class MADDPG:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device):
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device=device)
            self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim=action_dim if is_continue else 1,
                                            device=device)

        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0]  # sample用

    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            action = self.agents[agent_id].actor(obs)
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim -> action_dim
        return actions

    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        next_action = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(
                indices)
            next_action[agent_id] = self.agents[agent_id].actor_target(next_obs[agent_id])

        return obs, action, reward, next_obs, done, next_action

    def learn(self, batch_size, gamma, tau):
        # 多智能体特有-- 集中式训练critic
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
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_dir):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_dir, 'MADDPG.pth')
        )

    @staticmethod
    def load(dim_info, is_continue, model_dir):
        policy = MADDPG(dim_info, is_continue=is_continue, actor_lr=0, critic_lr=0, buffer_size=0, device='cpu')
        data = torch.load(os.path.join(model_dir, 'MADDPG.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return policy


## 第三部分：环境和训练配置
def get_env(env_name, render_mode=False):
    """创建UAV环境"""
    if env_name == 'uav_env':
        env = UAVEnv(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    env.reset()
    dim_info = {}  # dict{agent_id:[obs_dim, action_dim]}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)

    # UAV环境: action = [angle, comm_ch, jam_ch]
    # max_action: [2π, num_channels-1, num_channels-1]
    max_action = np.array([2 * math.pi, 9, 9])  # 根据配置，信道范围是[0-9]
    return env, dim_info, max_action, True  # is_continue = True


def make_dir(env_name, policy_name='MADDPG'):
    """创建保存模型的文件夹"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(script_dir, '../results', env_name)
    os.makedirs(env_dir, exist_ok=True)

    # 查找现有的文件夹并确定下一个编号
    prefix = policy_name + '_'
    pattern = re.compile(f'^{prefix}\d+')
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max(
        [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type=str, default="uav_env")
    # 共有参数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=int(100000))
    parser.add_argument("--save_freq", type=int, default=int(10000))
    parser.add_argument("--start_steps", type=int, default=5000)  # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=5000)  # 满足此开始自己探索
    parser.add_argument("--learn_interval", type=int, default=1)  # 每个episode学习的间隔
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=1024)
    # 噪声参数
    parser.add_argument("--gauss_sigma", type=float, default=0.1)  # 高斯噪声标准差
    parser.add_argument("--gauss_scale", type=float, default=1.0)
    # 策略名称
    parser.add_argument("--policy_name", type=str, default='MADDPG_0')
    # device参数
    parser.add_argument("--device", type=str, default='cpu')  # cpu/cuda
    # UAV环境参数
    parser.add_argument('--policy_number', type=int, default=0, help='蓝方策略编号')

    args = parser.parse_args()
    print(args)
    print('-' * 50)
    print('Algorithm:', args.policy_name)

    ## 环境配置
    env, dim_info, max_action, is_continue = get_env(args.env_name)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:', args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name, policy_name=args.policy_name)
    writer = SummaryWriter(model_dir)
    print('model_dir:', model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    ## 算法配置
    policy = MADDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device)

    time_ = time.time()
    ## 训练统计
    win_list = []  # 胜率
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}

    obs, infos = env.reset(args.policy_number)
    # obs已经在env.reset()中归一化了

    while episode_num < args.max_episodes:
        step += 1

        # 获取动作
        if step < args.random_steps:
            action = env.sample()  # 随机探索
        else:
            action = policy.select_action(obs)

        # 添加噪声（3维动作空间，分别加噪声）
        action_ = {}
        for agent_id in env_agents:
            # action: [-1, 1], 需要映射到实际范围
            # action[agent_id] 是 [angle_norm, comm_ch_norm, jam_ch_norm]
            noise = args.gauss_scale * np.random.normal(scale=args.gauss_sigma, size=dim_info[agent_id][1])
            action_with_noise = action[agent_id] + noise

            # 映射到实际范围: [-1,1] -> [0, max_action]
            # angle: [-1,1] -> [0, 2π]
            # comm_ch: [-1,1] -> [0, 9]
            # jam_ch: [-1,1] -> [0, 9]
            action_actual = (action_with_noise + 1) / 2 * max_action  # [-1,1] -> [0, max_action]
            action_actual = np.clip(action_actual, 0, max_action)
            action_[agent_id] = action_actual

        # 探索环境
        next_obs, reward, terminated, truncated, infos = env.step(action_)
        # next_obs已经在env.step()中归一化了

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        # 简化的奖励结算
        if any(done.values()):
            win_list.append(1 if infos["win"] else 0)
            # 胜利额外奖励
            if infos['win']:
                for agent_id in env_agents:
                    reward[agent_id] += 5.0  # 胜利奖励
                    if infos[agent_id] > 1e-3:  # 存活
                        reward[agent_id] += 2.0

        policy.add(obs, action, reward, next_obs, done_bool)
        for agent_id, r in reward.items():
            episode_reward[agent_id] += r
        obs = next_obs

        # episode结束
        if any(done.values()):
            ## 显示
            if (episode_num + 1) % 100 == 0:
                message = f'episode {episode_num + 1}, '
                win_rate = np.mean(win_list[-100:])
                writer.add_scalar('win_rate', win_rate, episode_num)

                # 显示UAV状态
                message += f'red_hp:'
                for agent_id in env.agents:
                    message += f'{infos[agent_id]:.1f},'
                message = message[:-1] + ';'

                message += f'blue_hp:'
                for agent_id in env.agents_e:
                    message += f'{infos[agent_id]:.1f},'
                message = message[:-1] + ';'

                message += f'win rate: {win_rate:.2f}; '
                print(message)

            for agent_id, r in episode_reward.items():
                writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                train_return[agent_id].append(r)

            episode_num += 1
            obs, infos = env.reset(args.policy_number)
            episode_reward = {agent_id: 0 for agent_id in env_agents}

            # 满足step，更新网络
            if step > args.start_steps and episode_num % args.learn_interval == 0:
                policy.learn(args.batch_size, args.gamma, args.tau)

        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    print('total_time:', time.time() - time_)
    policy.save(model_dir)

    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}.npy"), train_return_)
    win_list = np.array(win_list)
    np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}_win.npy"), win_list)
