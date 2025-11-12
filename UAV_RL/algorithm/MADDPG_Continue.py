"""
从已有模型继续训练（迁移学习）
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import math
from types import SimpleNamespace
from env_uav import UAVEnv
from UAV_RL.algorithm.MADDPG import MADDPG
from UAV_RL.utils.common import make_dir
from torch.utils.tensorboard import SummaryWriter
import time

def continue_training(pretrained_model_dir, target_policy, max_episodes=50000):
    """
    从预训练模型继续训练

    Args:
        pretrained_model_dir: 预训练模型路径（例如：对策略0训练的模型）
        target_policy: 目标策略编号（例如：1=规则策略）
        max_episodes: 继续训练的最大episodes
    """
    # 训练配置
    args = SimpleNamespace(
        env_name='uav_env',
        seed=0,
        max_episodes=max_episodes,
        save_freq=int(5000),  # 更频繁保存
        start_steps=0,  # 立即开始学习（不需要收集经验）
        random_steps=0,  # 立即使用策略（不随机探索）
        learn_interval=1,
        gamma=0.95,
        tau=0.01,
        actor_lr=1e-4,  # 更小的学习率（微调）
        critic_lr=1e-4,
        buffer_size=int(1e6),
        batch_size=512,
        gauss_sigma=0.2,  # 更大的探索噪声
        gauss_scale=1.0,
        policy_name=f'MADDPG_Continue_Policy{target_policy}',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_number=target_policy,
    )

    print("="*60)
    print("迁移学习：从预训练模型继续训练")
    print("="*60)
    print(f"预训练模型: {pretrained_model_dir}")
    print(f"目标策略: {target_policy}")
    print(f"训练episodes: {max_episodes}")
    print("="*60)

    # 创建环境
    env = UAVEnv()
    env.reset()

    # 获取维度信息
    dim_info = {}
    for agent_id in env.agents:
        obs_dim = env.observation_space(agent_id).shape[0]
        action_dim = env.action_space(agent_id).shape[0]
        dim_info[agent_id] = [obs_dim, action_dim]

    # 获取动作空间上限
    act_space = env.action_space(env.agents[0])
    num_channels_minus1 = int(act_space.high[1])
    max_action = np.array([2 * math.pi, num_channels_minus1, num_channels_minus1], dtype=np.float32)

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建保存目录
    model_dir = make_dir(args.env_name, policy_name=args.policy_name)
    writer = SummaryWriter(model_dir)
    print(f'Model directory: {model_dir}')

    # 加载预训练模型
    print(f"\n加载预训练模型...")
    device = torch.device(args.device)
    policy = MADDPG.load(dim_info, True, pretrained_model_dir)

    # 重新设置优化器（用于继续训练）
    for agent in policy.agents.values():
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr)
        agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=args.critic_lr)

    # 将模型移到正确的device
    for agent in policy.agents.values():
        agent.actor = agent.actor.to(device)
        agent.critic = agent.critic.to(device)
        agent.actor_target = agent.actor_target.to(device)
        agent.critic_target = agent.critic_target.to(device)

    policy.device = device
    print("预训练模型加载成功！")

    # 训练统计
    win_list = []
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}

    obs, infos = env.reset(args.policy_number)
    start_time = time.time()

    print(f"\n开始训练...")
    while episode_num < args.max_episodes:
        step += 1

        # 使用策略选择动作（带探索噪声）
        action = policy.select_action(obs)

        # 添加噪声并映射
        action_ = {}
        for agent_id in env_agents:
            noise = args.gauss_scale * np.random.normal(scale=args.gauss_sigma, size=dim_info[agent_id][1])
            action_with_noise = action[agent_id] + noise

            angle = np.clip(action_with_noise[0], -1.0, 1.0) * (2 * math.pi)
            comm = np.clip((action_with_noise[1] + 1.0) / 2.0 * max_action[1], 0.0, max_action[1])
            jam = np.clip((action_with_noise[2] + 1.0) / 2.0 * max_action[2], 0.0, max_action[2])

            action_[agent_id] = np.array([angle, comm, jam], dtype=np.float32)

        # 执行动作
        next_obs, reward, terminated, truncated, infos = env.step(action_)

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        # 奖励调整
        if any(done.values()):
            win_list.append(1 if infos["win"] else 0)
            if infos['win']:
                for agent_id in env_agents:
                    reward[agent_id] += 5.0
                    if infos[agent_id] > 1e-3:
                        reward[agent_id] += 2.0

        policy.add(obs, action, reward, next_obs, done_bool)
        for agent_id, r in reward.items():
            episode_reward[agent_id] += r
        obs = next_obs

        # episode结束
        if any(done.values()):
            if (episode_num + 1) % 100 == 0:
                win_rate = np.mean(win_list[-100:])
                writer.add_scalar('win_rate', win_rate, episode_num)

                message = f'episode {episode_num + 1}, '
                message += f'red_hp:'
                for agent_id in env.agents:
                    message += f'{infos[agent_id]:.1f},'
                message = message[:-1] + '; '
                message += f'blue_hp:'
                for agent_id in env.agents_e:
                    message += f'{infos[agent_id]:.1f},'
                message = message[:-1] + '; '
                message += f'win rate: {win_rate:.2f}'
                print(message)

            for agent_id, r in episode_reward.items():
                writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                train_return[agent_id].append(r)

            episode_num += 1
            obs, infos = env.reset(args.policy_number)
            episode_reward = {agent_id: 0 for agent_id in env_agents}

            # 学习
            if episode_num % args.learn_interval == 0:
                policy.learn(args.batch_size, args.gamma, args.tau)

        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    # 最终保存
    policy.save(model_dir)
    train_time = time.time() - start_time
    final_win_rate = np.mean(win_list[-1000:]) if len(win_list) >= 1000 else np.mean(win_list)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最终胜率: {final_win_rate:.2%}")
    print(f"总用时: {train_time/60:.1f}分钟")
    print(f"{'='*60}")

if __name__ == '__main__':
    # 修改为你的预训练模型路径
    pretrained_model_dir = r"d:\CodesFile\RedBlue\UAV_RL\results\uav_env\MADDPG_0_1"

    # 如果在Linux上，使用相对路径
    if not os.path.exists(pretrained_model_dir):
        pretrained_model_dir = os.path.join(
            os.path.dirname(__file__),
            '../results/uav_env/MADDPG_0_1'
        )

    if not os.path.exists(pretrained_model_dir):
        print(f"错误：预训练模型目录不存在: {pretrained_model_dir}")
        print("请修改 pretrained_model_dir 变量为正确的模型路径")
        sys.exit(1)

    # 对策略1继续训练
    continue_training(pretrained_model_dir, target_policy=1, max_episodes=50000)
