import gymnasium as gym
import math
from Tank_RL.env_tank_battle import ToyEnv

def get_env(env_name, render_mode=False):
    """
    获取环境并返回维度信息
    Args:
        env_name: 环境名称
        render_mode: 是否渲染
    Returns:
        env: 环境实例
        dim_info: 维度信息字典 {agent_id: [obs_dim, action_dim]}
        max_action: 最大动作值
        is_continue: 是否为连续动作空间
    """
    # 使用 Tank_RL 内部的环境
    if env_name == 'toy_env':
        env = ToyEnv(render_mode=render_mode)

    env.reset()
    dim_info = {}  # dict{agent_id:[obs_dim,action_dim]}
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

    return env, dim_info, 2 * math.pi, True
