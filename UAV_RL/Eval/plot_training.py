"""
可视化训练过程
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_results(model_dir, policy_name='MADDPG_0', seed=0):
    """
    绘制训练曲线

    Args:
        model_dir: 模型保存目录
        policy_name: 策略名称
        seed: 随机种子
    """
    # 加载数据
    win_file = os.path.join(model_dir, f"{policy_name}_seed_{seed}_win.npy")
    reward_file = os.path.join(model_dir, f"{policy_name}_seed_{seed}.npy")

    if not os.path.exists(win_file):
        print(f"错误：文件不存在 {win_file}")
        return

    win_data = np.load(win_file)  # shape: (num_episodes,)
    reward_data = np.load(reward_file)  # shape: (num_agents, num_episodes)

    num_episodes = len(win_data)
    print(f"加载数据：{num_episodes} episodes")

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ==================== 图1：胜率曲线 ====================
    ax1 = axes[0]

    # 计算滑动平均胜率
    window_size = 100
    win_rate = []
    for i in range(len(win_data)):
        start = max(0, i - window_size + 1)
        win_rate.append(np.mean(win_data[start:i+1]))

    ax1.plot(win_rate, linewidth=2, color='#2E86AB', label=f'Win Rate (avg over {window_size} eps)')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% Win Rate')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('Training Win Rate', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])

    # ==================== 图2：奖励曲线 ====================
    ax2 = axes[1]

    # 计算所有agent的平均奖励
    avg_reward = np.mean(reward_data, axis=0)  # 平均到所有agent

    # 计算滑动平均奖励
    reward_smooth = []
    for i in range(len(avg_reward)):
        start = max(0, i - window_size + 1)
        reward_smooth.append(np.mean(avg_reward[start:i+1]))

    ax2.plot(reward_smooth, linewidth=2, color='#A23B72', label=f'Avg Reward (avg over {window_size} eps)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Training Reward', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(model_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    plt.show()

    # ==================== 打印统计信息 ====================
    print("\n" + "="*50)
    print("训练统计")
    print("="*50)
    print(f"总Episodes: {num_episodes}")
    print(f"最终100回合平均胜率: {np.mean(win_data[-100:]):.2%}")
    print(f"最高100回合平均胜率: {max(win_rate[-1000:]):.2%}" if len(win_rate) > 1000 else f"最高胜率: {max(win_rate):.2%}")
    print(f"最终100回合平均奖励: {np.mean(avg_reward[-100:]):.2f}")
    print(f"最高平均奖励: {max(avg_reward):.2f}")
    print("="*50)


if __name__ == '__main__':
    # 修改为你的模型路径
    model_dir = r"d:\CodesFile\RedBlue\UAV_RL\results\uav_env\MADDPG_0_1"

    # 如果在Linux上，使用相对路径
    if not os.path.exists(model_dir):
        model_dir = os.path.join(
            os.path.dirname(__file__),
            '../results/uav_env/MADDPG_0_2'
        )

    plot_training_results(model_dir)
