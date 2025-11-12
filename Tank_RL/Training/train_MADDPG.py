import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import torch
import numpy as np
import time
import math
from torch.utils.tensorboard import SummaryWriter

from Tank_RL.algorithm import MADDPG
from Tank_RL.env_tank import get_env
from Tank_RL.utils import make_dir


# ==================== 配置参数 ====================
class Config:
    # 环境参数
    env_name = "tank_env"
    N = None  # 环境中智能体数量

    # 共有参数
    seed = 0
    max_episodes = 600000
    save_freq = 50000  # 200000 // 4
    start_steps = 50000  # 满足此开始更新
    random_steps = 50000  # 满足此开始自己探索
    learn_steps_interval = 1
    learn_interval = 3  # episode

    # 训练参数
    gamma = 0.95
    tau = 0.01

    # AC参数
    actor_lr = 1e-3
    critic_lr = 1e-3

    # buffer参数
    buffer_size = int(1e6)
    batch_size = 1024

    # DDPG 独有参数 noise
    gauss_sigma = 0.1  # 高斯标准差
    gauss_scale = 1
    gauss_init_scale = None  # 若不设置衰减，则设置成None
    gauss_final_scale = 0.0

    # trick参数
    policy_name = 'MADDPG_simple'
    trick = None

    # device参数
    device = 'cuda'  # cpu/cuda，优先使用cuda

    # 环境策略参数
    policy_number = 0
    policy_noise_a = 0.05 * 2 * math.pi
    policy_noise_b = 0.12 * 2 * math.pi
    win_rate_adjust = 0.03


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
    policy = MADDPG(dim_info, is_continue, cfg.actor_lr, cfg.critic_lr, cfg.buffer_size, device, cfg.trick)

    time_ = time.time()

    # 环境相关
    win_list = []  # 一般胜
    win1_list = []  # 大奖励（全灭敌人）
    win2_list = []  # 截断胜利（血量优势）
    red_tank_0_hp_l = []
    red_tank_1_hp_l = []
    red_tank_2_hp_l = []
    policy_up = 0

    # 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs, infos = env.reset(cfg.policy_number)
    obs = env.Normalization(obs)  # 状态归一化

    if cfg.gauss_init_scale is not None:
        cfg.gauss_scale = cfg.gauss_init_scale

    while episode_num < cfg.max_episodes:
        step += 1

        # 获取动作
        if step < cfg.random_steps:
            action = env.sample()
        else:
            action = policy.select_action(obs)

        # 加噪音
        action_ = {
            agent_id: np.clip(
                action[agent_id] * max_action + cfg.gauss_scale * np.random.normal(
                    scale=cfg.gauss_sigma * max_action,
                    size=dim_info[agent_id][1]
                ),
                -max_action,
                max_action
            )
            for agent_id in env_agents
        }

        # 探索环境
        next_obs, reward, terminated, truncated, infos = env.step(action_)
        next_obs = env.Normalization(next_obs)  # 状态归一化

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        if any(done.values()):
            win_list.append(1 if infos["win"] else 0)
            win1_list.append(1 if infos["win1"] else 0)
            win2_list.append(1 if infos["win2"] else 0)

            # 血量显示
            if infos["win1"]:
                red_tank_0_hp_l.append(infos["Red-tank-0"])
                red_tank_1_hp_l.append(infos["Red-tank-1"])
                red_tank_2_hp_l.append(infos["Red-tank-2"])

            # 结算奖励
            for agent_id, r in reward.items():
                if infos['win1'] == True:
                    reward[agent_id] += 10
                    if infos[agent_id] > 1e-3:
                        reward[agent_id] += 3  # 存活奖励
                        reward[agent_id] += infos[agent_id] * 3  # 生命值奖励

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

                if win1_rate >= cfg.win_rate_adjust:
                    policy_up += 1
                    cfg.gauss_sigma = cfg.gauss_sigma * 0.9
                    if win1_rate >= 0.10 or policy_up > 20:
                        cfg.win_rate_adjust += 0.02
                        episode_temp = episode_num
                    print(f'策略提升{policy_up}', cfg.gauss_sigma, cfg.win_rate_adjust)

                # 动态调整
                if win1_rate >= 0.10 or policy_up > 20:
                    if episode_num - episode_temp > 10000:
                        cfg.win_rate_adjust -= 0.01
                        cfg.gauss_sigma = cfg.gauss_sigma * 0.9
                        episode_temp = episode_num
                        print(f'策略调整{policy_up}adjust', cfg.gauss_sigma, cfg.win_rate_adjust)

                writer.add_scalar('win_rate', win_rate, episode_num)
                writer.add_scalar('win1_rate', win1_rate, episode_num)
                writer.add_scalar('win2_rate', win2_rate, episode_num)

                # 血量显示
                if len(red_tank_0_hp_l) > 100:
                    red_tank_0_hp = np.mean(red_tank_0_hp_l[-100:])
                    red_tank_1_hp = np.mean(red_tank_1_hp_l[-100:])
                    red_tank_2_hp = np.mean(red_tank_2_hp_l[-100:])
                    writer.add_scalar('red_tank_0_hp', red_tank_0_hp, episode_num)
                    writer.add_scalar('red_tank_1_hp', red_tank_1_hp, episode_num)
                    writer.add_scalar('red_tank_2_hp', red_tank_2_hp, episode_num)

                # print message
                message += f'red_hp:'
                for agend_id in env.agents:
                    message += f'{infos[agend_id]:.1f},'
                message = message[:-1] + ';'

                message += f'blue_hp:'
                for agend_id in env.agents_e:
                    message += f'{infos[agend_id]:.1f},'
                message = message[:-1] + ';'

                message += f'win rate: {win_rate:.2f}; '
                message += f'win1 rate: {win1_rate:.2f}; '
                print(message)

            for agent_id, r in episode_reward.items():
                writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                train_return[agent_id].append(r)

            episode_num += 1
            obs, infos = env.reset(cfg.policy_number)
            obs = env.Normalization(obs)
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
    if cfg.N is None:
        np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}.npy"), train_return_)
    else:
        np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_N_{len(env_agents)}.npy"), train_return_)

    # 保存win、win1 和 win2
    win_list = np.array(win_list)
    win1_list = np.array(win1_list)
    win2_list = np.array(win2_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win.npy"), win_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win1.npy"), win1_list)
    np.save(os.path.join(model_dir, f"{cfg.policy_name}_seed_{cfg.seed}_win2.npy"), win2_list)
