import numpy as np
import torch

'''
    这里将act_dim 和action_dim 区分开来
    1维离散空间 act_dim = 1  action_dim = 离散空间的维度       即 [0]
    3维离散空间 act_dim = 1  action_dim = 离散空间的维度 ** 3
    1维连续空间 act_dim = 1  action_dim = 1
    3维连续空间 act_dim = 3  action_dim = 3                   即 [0,0,0]
'''

class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity = int(capacity)

        self.obs = np.zeros((capacity, obs_dim))    # batch_size x state_dim
        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)            # just a tensor with length: batch
        self.next_obs = np.zeros((capacity, obs_dim))  # batch_size x state_dim
        self.dones = np.zeros(capacity, dtype=bool)    # just a tensor with length: batch_size

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)  # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1, 1).to(self.device)

        return obs, actions, rewards, next_obs, dones

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size
