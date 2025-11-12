"""
经验回放Buffer
用于MADDPG算法的离线学习
"""
import numpy as np
import torch


class Buffer:
    """
    经验回放Buffer（用于MADDPG）

    存储格式：
    - obs_dim: 观测维度（UAV环境为14维）
    - act_dim: 动作维度（UAV环境为3维：angle, comm_ch, jam_ch）

    支持：
    - 连续动作空间（act_dim = 动作维度）
    - 离散动作空间（act_dim = 1）
    """

    def __init__(self, capacity, obs_dim, act_dim, device):
        """
        初始化Buffer

        :param capacity: 经验池容量
        :param obs_dim: 观测维度
        :param act_dim: 动作维度（连续空间）或1（离散空间）
        :param device: 'cpu' 或 'cuda'
        """
        self.capacity = capacity = int(capacity)

        # 存储空间（使用numpy数组，节省内存）
        self.obs = np.zeros((capacity, obs_dim))        # 观测
        self.actions = np.zeros((capacity, act_dim))    # 动作
        self.rewards = np.zeros(capacity)               # 奖励
        self.next_obs = np.zeros((capacity, obs_dim))   # 下一个观测
        self.dones = np.zeros(capacity, dtype=bool)     # 是否终止

        self._index = 0   # 当前写入位置
        self._size = 0    # 当前buffer大小

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """
        添加一条经验到buffer

        :param obs: 观测
        :param action: 动作
        :param reward: 奖励
        :param next_obs: 下一个观测
        :param done: 是否终止
        """
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        # 循环队列：写满后从头开始覆盖
        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        """
        根据索引采样经验

        :param indices: 采样索引（numpy数组）
        :return: (obs, actions, rewards, next_obs, dones) 所有数据转换为torch.Tensor
        """
        # 提取数据（numpy数组）
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]

        # 转换为torch.Tensor并移动到指定device
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1).to(self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1, 1).to(self.device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        """返回当前buffer中的经验数量"""
        return self._size
