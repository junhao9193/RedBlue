import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import torch
import numpy as np
import time
import math
from torch.utils.tensorboard import SummaryWriter

from UAV_RL.algorithm.MADDPG import MADDPG
from UAV_RL.env_uav import get_env
from UAV_RL.utils.common import make_dir


# ==================== 配置参数 ====================
class Config:
    # 环境参数
    env_name = "uav_env"

    # 共有参数
    seed = 0
    max_episodes = 100000
    save_freq = 10000
    start_steps = 3000  # 满足此开始更新
    random_steps = 3000  # 满足此开始自己探索
    learn_interval = 1  # episode

    # 训练参数
    gamma = 0.95
    tau = 0.01

    # AC参数
    actor_lr = 3e-4
    critic_lr = 3e-4

    # buffer参数
    buffer_size = int(1e6)
    batch_size = 512

    # DDPG 独有参数 noise
    gauss_sigma = 0.15  # 高斯标准差
    gauss_scale = 1.0

    # trick参数
    policy_name = 'MADDPG_0'
    trick = None

    # device参数
    device = 'cuda'  # cpu/cuda，优先使用cuda

    # 环境策略参数
    policy_number = 1  # 蓝方对手策略编号（1=规则策略：攻击最近敌人）

    # 课程学习参数
    win_rate_adjust = 0.03  # 胜率阈值，初始3%


if __name__ == '__main__':
    # 加载配置
    cfg = Config()

    print(f'Training Configuration:')
    print(f'  Algorithm: {cfg.policy_name}')
    print(f'  Environment: {cfg.env_name}')
    print(f'  Max Episodes: {cfg.max_episodes}')
    print(f'  Device: {cfg.device}')
    print('-' * 50)

    # 环境配置
    env, dim_info, max_action, is_continue = get_env(cfg.env_name)
    print(f'Env: {cfg.env_name}')
    print(f'Dim info: {dim_info}')
    print(f'Max action: {max_action}')
    print(f'Max episodes: {cfg.max_episodes}')

    # 随机数种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Random Seed: {cfg.seed}')

    # 保存model文件夹
    model_dir = make_dir(cfg.env_name, policy_name=cfg.policy_name, trick=cfg.trick)
    writer = SummaryWriter(model_dir)
    print(f'Model dir: {model_dir}')

    # device参数：优先使用cuda，若不可用则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 算法配置
    policy = MADDPG(dim_info, is_continue, cfg.actor_lr, cfg.critic_lr, cfg.buffer_size, device)

    time_ = time.time()

    # 环境相关
    win_list = []  # 一般胜
    win1_list = []  # 大奖励（全灭敌人）
    win2_list = []  # 截断胜利（HP优势）
    policy_up = 0  # 策略提升计数器
    episode_temp = 0  # 记录上次调整的episode

    # 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs, infos = env.reset(cfg.policy_number)

    while episode_num < cfg.max_episodes:
        step += 1

        # 获取动作
        if step < cfg.random_steps:
            action = env.sample()  # 随机探索
        else:
            action = policy.select_action(obs)

        # 添加噪声并映射到环境动作空间
        # 角度需保留符号：angle<0 表示射击，>0 表示移动；信道映射到 [0, num_channels-1]
        # Actor 输出 a∈[-1,1]，映射规则：
        #   angle_actual = a_angle * 2π（保留正负号）
        #   comm_actual  = (a_comm + 1)/2 * (num_channels-1)
        #   jam_actual   = (a_jam  + 1)/2 * (num_channels-1)
        action_ = {}
        for agent_id in env_agents:
            noise = cfg.gauss_scale * np.random.normal(scale=cfg.gauss_sigma, size=dim_info[agent_id][1])
            action_with_noise = action[agent_id] + noise  # [-1,1] + 噪声

            # angle 映射到 [-2π, 2π]，保留正负号用于区分移动/射击
            angle = np.clip(action_with_noise[0], -1.0, 1.0) * (2 * math.pi)

            # 通道维度映射到 [0, num_channels-1]
            comm = np.clip((action_with_noise[1] + 1.0) / 2.0 * max_action[1], 0.0, max_action[1])
            jam = np.clip((action_with_noise[2] + 1.0) / 2.0 * max_action[2], 0.0, max_action[2])

            action_[agent_id] = np.array([angle, comm, jam], dtype=np.float32)

        # 探索环境
        next_obs, reward, terminated, truncated, infos = env.step(action_)

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        # 简化的奖励结算
        if any(done.values()):
            win_list.append(1 if infos["win"] else 0)
            win1_list.append(1 if infos["win1"] else 0)
            win2_list.append(1 if infos["win2"] else 0)

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

        # episode 结束
        if any(done.values()):
            # 显示
            if (episode_num + 1) % 100 == 0:
                message = f'episode {episode_num + 1}, '

                win_rate = np.mean(win_list[-100:])
                win1_rate = np.mean(win1_list[-100:])
                win2_rate = np.mean(win2_list[-100:])

                # ========== 自适应课程学习机制 ==========
                # 策略提升：当胜率达到阈值时，减少探索噪声并提高难度
                if win1_rate >= cfg.win_rate_adjust:
                    policy_up += 1
                    cfg.gauss_sigma = cfg.gauss_sigma * 0.9  # 逐步降低探索噪声

                    if win1_rate >= 0.10 or policy_up > 20:
                        cfg.win_rate_adjust += 0.02  # 提高胜率阈值，增加难度
                        episode_temp = episode_num
                    print(f'策略提升{policy_up}, gauss_sigma={cfg.gauss_sigma:.4f}, win_rate_adjust={cfg.win_rate_adjust:.2f}')

                # 动态调整：如果长时间没有进步，降低难度避免训练卡死
                if win1_rate >= 0.10 or policy_up > 20:
                    if episode_num - episode_temp > 10000:
                        cfg.win_rate_adjust -= 0.01  # 降低胜率阈值
                        cfg.gauss_sigma = cfg.gauss_sigma * 0.9  # 继续降低噪声
                        episode_temp = episode_num
                        print(f'策略调整{policy_up}adjust, gauss_sigma={cfg.gauss_sigma:.4f}, win_rate_adjust={cfg.win_rate_adjust:.2f}')
                # ========================================

                writer.add_scalar('win_rate', win_rate, episode_num)
                writer.add_scalar('win1_rate', win1_rate, episode_num)
                writer.add_scalar('win2_rate', win2_rate, episode_num)
                writer.add_scalar('gauss_sigma', cfg.gauss_sigma, episode_num)  # 记录噪声变化

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
                message += f'win1 rate: {win1_rate:.2f}; '
                message += f'noise: {cfg.gauss_sigma:.4f}; '
                print(message)

            for agent_id, r in episode_reward.items():
                writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                train_return[agent_id].append(r)

            episode_num += 1
            obs, infos = env.reset(cfg.policy_number)
            episode_reward = {agent_id: 0 for agent_id in env_agents}

            # 满足step,更新网络
            if step > cfg.start_steps and episode_num % cfg.learn_interval == 0:
                policy.learn(cfg.batch_size, cfg.gamma, cfg.tau)

        # 保存模型
        if episode_num % cfg.save_freq == 0:
            policy.save(model_dir)

    print(f'Total time: {time.time() - time_:.2f}s')
    policy.save(model_dir)

    # 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}.npy"), train_return_)

    # 保存win、win1 和 win2
    win_list = np.array(win_list)
    win1_list = np.array(win1_list)
    win2_list = np.array(win2_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win.npy"), win_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win1.npy"), win1_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win2.npy"), win2_list)
