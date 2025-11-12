"""
课程学习训练脚本 - 逐步提高蓝方难度
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
from MADDPG_0 import MADDPG, make_dir
from torch.utils.tensorboard import SummaryWriter
import time

def curriculum_learning_train():
    """
    课程学习训练流程：
    阶段1：对策略0训练 20000 episodes（从头开始）
    阶段2：对策略1训练 30000 episodes（基于阶段1模型）
    阶段3：对策略3训练 50000 episodes（基于阶段2模型）
    """

    # 训练配置
    args = SimpleNamespace(
        env_name='uav_env',
        seed=0,
        save_freq=int(10000),
        start_steps=3000,
        random_steps=3000,
        learn_interval=1,
        gamma=0.95,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=3e-4,
        buffer_size=int(1e6),
        batch_size=512,
        gauss_sigma=0.15,
        gauss_scale=1.0,
        policy_name='MADDPG_Curriculum',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # 课程设置
    curriculum = [
        {'policy': 0, 'episodes': 20000, 'name': '策略0-随机'},
        {'policy': 1, 'episodes': 30000, 'name': '策略1-规则'},
        {'policy': 3, 'episodes': 50000, 'name': '策略3-混合'},
    ]

    print("="*60)
    print("课程学习训练")
    print("="*60)
    for i, stage in enumerate(curriculum):
        print(f"阶段{i+1}: {stage['name']} - {stage['episodes']} episodes")
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
    torch.cuda.manual_seed(args.seed)

    # 创建保存目录
    model_dir = make_dir(args.env_name, policy_name=args.policy_name)
    writer = SummaryWriter(model_dir)
    print(f'Model directory: {model_dir}')

    # 创建device
    device = torch.device(args.device)

    # 初始化MADDPG
    policy = MADDPG(dim_info, True, args.actor_lr, args.critic_lr, args.buffer_size, device)

    # 训练统计
    total_episodes = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]

    # 开始课程学习
    for stage_idx, stage in enumerate(curriculum):
        policy_num = stage['policy']
        stage_episodes = stage['episodes']
        stage_name = stage['name']

        print(f"\n{'='*60}")
        print(f"开始阶段{stage_idx+1}: {stage_name}")
        print(f"对手策略: {policy_num}, 训练episodes: {stage_episodes}")
        print(f"{'='*60}\n")

        # 阶段内训练
        episode_num = 0
        win_list = []
        episode_reward = {agent_id: 0 for agent_id in env_agents}
        train_return = {agent_id: [] for agent_id in env_agents}

        obs, infos = env.reset(policy_num)
        stage_start_time = time.time()

        while episode_num < stage_episodes:
            step += 1

            # 获取动作
            if step < args.random_steps:
                action = env.sample()
            else:
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
                    message = f'Stage {stage_idx+1}, Episode {episode_num + 1}/{stage_episodes}, '
                    message += f'Total {total_episodes + episode_num + 1}, '
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

                # 记录到TensorBoard
                writer.add_scalar(f'Stage{stage_idx+1}/win_rate', np.mean(win_list[-100:]) if len(win_list) >= 100 else 0,
                                  total_episodes + episode_num)

                for agent_id, r in episode_reward.items():
                    writer.add_scalar(f'Stage{stage_idx+1}/reward_{agent_id}', r, total_episodes + episode_num + 1)
                    train_return[agent_id].append(r)

                episode_num += 1
                obs, infos = env.reset(policy_num)
                episode_reward = {agent_id: 0 for agent_id in env_agents}

                # 学习
                if step > args.start_steps and episode_num % args.learn_interval == 0:
                    policy.learn(args.batch_size, args.gamma, args.tau)

            # 保存模型
            if (total_episodes + episode_num) % args.save_freq == 0:
                policy.save(model_dir)

        # 阶段完成
        total_episodes += stage_episodes
        stage_time = time.time() - stage_start_time
        stage_win_rate = np.mean(win_list[-1000:]) if len(win_list) >= 1000 else np.mean(win_list)

        print(f"\n{'='*60}")
        print(f"阶段{stage_idx+1}完成!")
        print(f"最终胜率: {stage_win_rate:.2%}")
        print(f"用时: {stage_time/60:.1f}分钟")
        print(f"{'='*60}\n")

        # 保存阶段模型
        stage_model_dir = os.path.join(model_dir, f'stage{stage_idx+1}_policy{policy_num}')
        os.makedirs(stage_model_dir, exist_ok=True)
        policy.save(stage_model_dir)

    # 最终保存
    policy.save(model_dir)
    print(f"\n课程学习训练完成！总episodes: {total_episodes}")

if __name__ == '__main__':
    curriculum_learning_train()
