"""
测试训练好的MADDPG模型
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import math
from env_uav import UAVEnv
from algorithm.MADDPG_0 import MADDPG

def test_policy(model_dir, policy_number=1, num_episodes=100):
    """
    测试训练好的策略

    Args:
        model_dir: 模型保存路径
        policy_number: 蓝方策略编号（0=随机，1=规则，3=混合）
        num_episodes: 测试回合数
    """
    # 创建环境
    env = UAVEnv(render_mode=False)

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

    # 加载模型
    print(f"Loading model from: {model_dir}")
    policy = MADDPG.load(dim_info, is_continue=True, model_dir=model_dir)

    # 测试统计
    win_count = 0
    red_hp_total = []
    blue_hp_total = []

    for episode in range(num_episodes):
        obs, infos = env.reset(policy_number)
        episode_done = False
        step_count = 0

        while not episode_done:
            # 使用策略选择动作（无噪声）
            actions = policy.select_action(obs)

            # 映射动作到环境空间
            action_ = {}
            for agent_id in env.agents:
                action_tensor = actions[agent_id]

                # angle 映射到 [-2π, 2π]
                angle = np.clip(action_tensor[0], -1.0, 1.0) * (2 * math.pi)
                # 信道映射到 [0, num_channels-1]
                comm = np.clip((action_tensor[1] + 1.0) / 2.0 * max_action[1], 0.0, max_action[1])
                jam = np.clip((action_tensor[2] + 1.0) / 2.0 * max_action[2], 0.0, max_action[2])

                action_[agent_id] = np.array([angle, comm, jam], dtype=np.float32)

            # 执行动作
            obs, reward, terminated, truncated, infos = env.step(action_)

            step_count += 1
            done = any(terminated.values()) or any(truncated.values())

            if done:
                episode_done = True

                # 统计结果
                if infos.get('win', False):
                    win_count += 1

                # 记录HP
                red_hp = sum([infos[agent_id] for agent_id in env.agents])
                blue_hp = sum([infos[agent_id] for agent_id in env.agents_e])
                red_hp_total.append(red_hp)
                blue_hp_total.append(blue_hp)

                # 打印进度
                if (episode + 1) % 10 == 0:
                    current_win_rate = win_count / (episode + 1)
                    print(f"Episode {episode + 1}/{num_episodes} - Win Rate: {current_win_rate:.2%}")

    # 最终统计
    win_rate = win_count / num_episodes
    avg_red_hp = np.mean(red_hp_total)
    avg_blue_hp = np.mean(blue_hp_total)

    print("\n" + "="*50)
    print(f"测试完成！")
    print(f"对手策略: {policy_number} ({'随机' if policy_number==0 else '规则' if policy_number==1 else '混合'})")
    print(f"总回合数: {num_episodes}")
    print(f"胜场数: {win_count}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均红方剩余HP: {avg_red_hp:.2f}")
    print(f"平均蓝方剩余HP: {avg_blue_hp:.2f}")
    print("="*50)

    return win_rate, avg_red_hp, avg_blue_hp


if __name__ == '__main__':
    # 使用相对路径（从Eval文件夹向上一级，再到results）
    model_dir = os.path.join(
        os.path.dirname(__file__),
        '../results/uav_env/MADDPG_0_1'
    )
    # 如果相对路径不存在，尝试绝对路径
    if not os.path.exists(model_dir):
        model_dir = r"d:\CodesFile\RedBlue\UAV_RL\results\uav_env\MADDPG_0_1"

    if not os.path.exists(model_dir):
        print(f"错误：模型目录不存在: {model_dir}")
        print("请修改 model_dir 变量为正确的模型路径")
        sys.exit(1)

    print("="*50)
    print("MADDPG模型测试")
    print("="*50)

    # 测试对抗策略0（随机）
    print("\n[测试1] 对抗策略0（随机策略）")
    test_policy(model_dir, policy_number=0, num_episodes=100)

    # 测试对抗策略1（规则）
    print("\n[测试2] 对抗策略1（规则策略）")
    test_policy(model_dir, policy_number=1, num_episodes=100)

    # 测试对抗策略3（混合）
    print("\n[测试3] 对抗策略3（随机混合策略）")
    test_policy(model_dir, policy_number=3, num_episodes=100)
