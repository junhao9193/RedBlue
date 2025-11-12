import os
import re

def make_dir(env_name, policy_name='DQN', trick=None):
    """
    创建模型保存目录
    Args:
        env_name: 环境名称
        policy_name: 策略名称
        trick: trick字典
    Returns:
        model_dir: 模型保存路径
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本文件夹
    # 向上两级到达 Tank_RL 的父目录，然后进入 Tank_RL/results
    env_dir = os.path.join(script_dir, '..', 'results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:', trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    pattern = re.compile(f'^{prefix}\d+')  # ^ 表示开头，\d 表示数字，+表示至少一个
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir
