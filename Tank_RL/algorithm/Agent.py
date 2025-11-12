import torch
from copy import deepcopy
from Tank_RL.models import Actor, Critic

class Agent:
    """
    单个智能体类，包含 Actor 和 Critic 网络及其优化器
    """
    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, device):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(dim_info).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
