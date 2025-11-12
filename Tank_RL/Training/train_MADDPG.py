import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'  # 确保在pytorch前设置

import torch
import numpy as np
import argparse
import time
import math
from torch.utils.tensorboard import SummaryWriter

from Tank_RL.algorithm import MADDPG
from Tank_RL.env_tank import get_env
from Tank_RL.utils import make_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type=str, default="toy_env")
    parser.add_argument("--N", type=int, default=None)  # 环境中智能体数量 默认None 这里用来对比设置
    # 共有参数
    parser.add_argument("--seed", type=int, default=0)  # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(600000))
    parser.add_argument("--save_freq", type=int, default=int(200000 // 4))
    parser.add_argument("--start_steps", type=int, default=50000)  # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=50000)  # 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--learn_interval", type=int, default=3)  # episode
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数
    parser.add_argument("--buffer_size", type=int, default=1e6)  # 1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=1024)  # 保证比start_steps小
    # DDPG 独有参数 noise
    parser.add_argument("--gauss_sigma", type=float, default=0.1)  # 高斯标准差
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=None)  # 若不设置衰减，则设置成None
    parser.add_argument("--gauss_final_scale", type=float, default=0.0)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MADDPG_simple')
    parser.add_argument("--trick", type=dict, default=None)
    # device参数
    parser.add_argument("--device", type=str, default='cpu')  # cpu/cuda

    # 环境参数
    parser.add_argument('--policy_number', type=int, default=0, help='number of policy')
    parser.add_argument('--policy_noise_a', type=float, default=0.05 * 2 * math.pi, help='policy noise 1')
    parser.add_argument('--policy_noise_b', type=float, default=0.12 * 2 * math.pi, help='policy noise 1')
    parser.add_argument('--win_rate_adjust', type=float, default=0.03, help='win rate adjust')

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
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:', args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name, policy_name=args.policy_name, trick=args.trick)
    writer = SummaryWriter(model_dir)
    print('model_dir:', model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    ## 算法配置
    policy = MADDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

    time_ = time.time()
    # 环境相关
    win_list = []  # 一般胜
    win1_list = []  # 大奖励

    red_tank_0_hp_l = []
    red_tank_1_hp_l = []
    red_tank_2_hp_l = []
    policy_up = 0

    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs, infos = env.reset(args.policy_number)
    obs = env.Normalization(obs)  # 状态归一
    if args.gauss_init_scale is not None:
        args.gauss_scale = args.gauss_init_scale
    while episode_num < args.max_episodes:
        step += 1

        # 获取动作
        if step < args.random_steps:
            action = env.sample()
        else:
            action = policy.select_action(obs)
            # 加噪音
        action_ = {agent_id: np.clip(action[agent_id] * max_action + args.gauss_scale * np.random.normal(scale=args.gauss_sigma * max_action, size=dim_info[agent_id][1]), -max_action, max_action) for agent_id in env_agents}

        # 探索环境
        next_obs, reward, terminated, truncated, infos = env.step(action_)
        next_obs = env.Normalization(next_obs)  # 状态归一

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}  ### truncated 为超过最大步数

        if any(done.values()):
            win_list.append(1 if infos["win"] else 0)
            win1_list.append(1 if infos["win1"] else 0)
            # 血量显示
            if infos["win1"]:
                red_tank_0_hp_l.append(infos["Red-tank-0"])
                red_tank_1_hp_l.append(infos["Red-tank-1"])
                red_tank_2_hp_l.append(infos["Red-tank-2"])
            # 结算奖励
            for agent_id, r in reward.items():
                if infos['win1'] == True:  # 只有结算时有 大win
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
            ## 显示
            if (episode_num + 1) % 100 == 0:
                message = f'episode {episode_num + 1}, '

                win_rate = np.mean(win_list[-100:])  # 一般胜率
                win1_rate = np.mean(win1_list[-100:])
                if win1_rate >= args.win_rate_adjust:
                    policy_up += 1
                    args.gauss_sigma = args.gauss_sigma * 0.9
                    if win1_rate >= 0.10 or policy_up > 20:
                        args.win_rate_adjust += 0.02
                        episode_temp = episode_num
                    print(f'策略提升{policy_up}', args.gauss_sigma, args.win_rate_adjust)
                # 动态调整
                if win1_rate >= 0.10 or policy_up > 20:  # 此时episode_temp才有定义
                    if episode_num - episode_temp > 10000:
                        args.win_rate_adjust -= 0.01
                        args.gauss_sigma = args.gauss_sigma * 0.9
                        episode_temp = episode_num
                        print(f'策略调整{policy_up}adjust', args.gauss_sigma, args.win_rate_adjust)
                writer.add_scalar('win_rate', win_rate, episode_num)
                writer.add_scalar('win1_rate', win1_rate, episode_num)  # 大胜利
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
            obs, infos = env.reset(args.policy_number)
            obs = env.Normalization(obs)  # 状态归一
            episode_reward = {agent_id: 0 for agent_id in env_agents}

            # 满足step,更新网络
            if step > args.start_steps and episode_num % args.learn_interval == 0:
                policy.learn(args.batch_size, args.gamma, args.tau)

        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    print('total_time:', time.time() - time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}.npy"), train_return_)
    else:
        np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"), train_return_)
    ### 保存win1 和win
    win1_list = np.array(win1_list)
    win_list = np.array(win_list)
    np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}_win1.npy"), win1_list)
    np.save(os.path.join(model_dir, f"{args.policy_name}_seed_{args.seed}_win.npy"), win_list)
