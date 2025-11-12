import os
import torch
import torch.nn.functional as F
import numpy as np
from Tank_RL.algorithm.Agent import Agent
from Tank_RL.utils.Buffer import Buffer

'''maddpg
论文：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments 链接：https://arxiv.org/abs/1706.02275
更新代码：https://github.com/openai/maddpg
原始代码:https://www.dropbox.com/scl/fi/2qb2470nj60qk7wb10y2s/maddpg_ensemble_and_approx_code.zip
创新点(特点--与COMA不同点)
1.为每个agent学习一个集中式critic 允许agent具有不同奖励
2.考虑了具有明确agent之间通信的环境
3.只使用前馈网络 不使用循环网络
4.学习连续策略
缺点：Q 的输入空间随着agent_N的数量呈线性增长,
展望：通过一个模块化的Q来修复，该函数只考虑该agent的某个领域的几个代理

可参考参数：
hidden：64 - 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
buffer_size = 1e6
batch_size = 1024
tau = 0.01
'''

# 此simple为选择MADDPG_reproduction的actor_learn_way=0的版本
class MADDPG:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick=None):
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device=device)
            self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim=action_dim if is_continue else 1, device=device)

        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0]  # sample 用

        self.regular = False  # 与DDPG中使用的weight_decay原理一致

    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            # if self.is_continue: # 现仅实现continue
            action = self.agents[agent_id].actor(obs)
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim -> action_dim
            # else:
            #     action = self.agents[agent_id].argmax(dim = 1).detach().cpu().numpy()[0] # []标量
            #     actions[agent_id] = action
        return actions

    def evaluate_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:  # 现仅实现continue
                action = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim -> action_dim
        return actions

    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        next_action = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)
            next_action[agent_id] = self.agents[agent_id].actor_target(next_obs[agent_id])

        return obs, action, reward, next_obs, done, next_action  # 包含所有智能体的数据

    ## DDPG算法相关
    '''论文中提出两种方法更新actor 最终论文取了方式0作为伪代码 论文中比较使用方式0,1 发现方式0的学习曲线效果与方式1比稍差略微，但KL散度差于方式1
    0. actor_loss = -critic(x, actor(obs),other_act).mean()    知道其他agent的策略来更新             此时next_target_Q = agent.critic_target(next_obs.values(), next_action.values())
    1. actor_loss = -(log(actor(obs)) + lmbda * H(actor_dist)) 知道其他智能体的obs但不知道策略来更新  此时next_target_Q 与上述一样
    这里选择0实现。
    '''

    def learn(self, batch_size, gamma, tau):
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            obs, action, reward, next_obs, done, next_action = self.sample(batch_size)  # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            next_target_Q = agent.critic_target(next_obs.values(), next_action.values())

            # 先更新critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            # 再更新actor
            new_action = agent.actor(obs[agent_id])
            action[agent_id] = new_action
            actor_loss = -agent.critic(obs.values(), action.values()).mean()
            # if self.regular : # 和DDPG.py中的weight_decay一样原理
            #     actor_loss += (new_action**2).mean() * 1e-3
            agent.update_actor(actor_loss)

        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_dir):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_dir, 'MADDPG.pth')
        )

    ## 加载模型
    @staticmethod
    def load(dim_info, is_continue, model_dir, trick=None):
        policy = MADDPG(dim_info, is_continue=is_continue, actor_lr=0, critic_lr=0, buffer_size=0, device='cpu')
        data = torch.load(os.path.join(model_dir, 'MADDPG.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return policy
