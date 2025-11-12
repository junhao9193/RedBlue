"""通用工具函数"""
import os
import re


def make_dir(env_name, policy_name='MADDPG', trick=None):
    """创建保存模型的文件夹

    自动创建带编号的目录，避免覆盖已有结果

    Args:
        env_name: 环境名称
        policy_name: 策略名称
        trick: 技巧名称（可选）

    Returns:
        model_dir: 创建的目录路径
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(script_dir, '../results', env_name)
    os.makedirs(env_dir, exist_ok=True)

    # 如果有trick，添加到policy_name后面
    if trick:
        policy_name = f"{policy_name}_{trick}"

    # 查找现有的文件夹并确定下一个编号
    prefix = policy_name + '_'
    pattern = re.compile(f'^{re.escape(prefix)}\\d+$')
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]

    max_number = 0
    if existing_dirs:
        numbers = []
        for d in existing_dirs:
            num_str = d.replace(prefix, '')
            if num_str.isdigit():
                numbers.append(int(num_str))
        if numbers:
            max_number = max(numbers)

    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir
