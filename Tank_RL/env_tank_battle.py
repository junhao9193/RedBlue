"""3v3坦克对抗环境实现 - 使用 Tank_RL 配置和 pojo"""
import math
import numpy as np
from collections import deque
from pathlib import Path
import addict
import toml
from gymnasium import spaces

# 导入 Tank_RL 的 pojo 类
from Tank_RL.pojo import Tank, Bullet
from Tank_RL.mytyping import Action, Info, Number, Observation, Reward, TeamName

# 读取 Tank_RL 配置
config = addict.Dict(toml.load(Path(__file__).parent / "config" / "tank.toml"))
config.observation_space = {
    "x": [0, config.map.width],
    "y": [0, config.map.height],
    "hp_t": [0, config.tank.hp],
    "speed_t": [0, config.tank.speed],
    "bullet_t": [0, config.tank.bullet_num],
    "fuel": [0, config.tank.fuel],
}

__all__ = ["ToyEnv", "config"]


class Team:
    """队伍类实现，一共可能有两种类型：红队或蓝队"""

    def __init__(self, name: TeamName) -> None:
        assert name in TeamName.__args__
        self.name = name
        _y = config.map.height if self.name == "Blue" else 0

        # 坦克
        self.tanks = [
            Tank(
                config.map.width / 2 + config.map.width / 5 * ((i + 1) // 2) * (-1) ** i,
                _y,
                self.name,
            )
            for i in range(config.team.get("n_{}_tank".format(self.name.lower())))
        ]

    @property
    def alives(self) -> list[bool]:
        """获取队伍所有坦克的存活状态"""
        return [t.alive for t in self.tanks]


class ToyEnv:
    """简易坦克对战环境"""

    def __init__(self, render_mode=False) -> None:
        self.render_mode = render_mode
        self.step_size = 30 / config.max_steps
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0
        self.agents = [i.uid for i in self.red_team.tanks]
        self.agents_e = [i.uid for i in self.blue_team.tanks]
        self.episode_length = config.max_steps

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def tanks(self) -> list[Tank]:
        """所有的坦克实例"""
        return self.red_team.tanks + self.blue_team.tanks

    @property
    def red_tanks(self) -> list[Tank]:
        """红队的坦克实例"""
        return self.red_team.tanks

    @property
    def blue_tanks(self) -> list[Tank]:
        """蓝队的坦克实例"""
        return self.blue_team.tanks

    @staticmethod
    def compute_distance(*args) -> float:
        if len(args) == 2:
            o1, o2 = args
            d = math.sqrt((o1.x - o2.x) ** 2 + (o1.y - o2.y) ** 2)
        else:
            o1, o2, o3 = args
            a = o2.y - o1.y
            b = o1.x - o2.x
            c = o2.x * o1.y - o1.x * o2.y
            d = (a * o3.x + b * o3.y + c) / math.sqrt((a**2 + b**2 + 1e-3))
        return d

    @staticmethod
    def atan(o1, o2) -> float:
        theta = math.atan2(o2.y - o1.y, o2.x - o1.x)
        if theta < 0:
            theta += 2 * math.pi
        return theta

    def sample(self) -> dict[str, Action]:
        """随机生成动作"""
        actions = {}
        for t in self.red_tanks:
            actions[t.uid] = np.array([np.random.uniform(-1, 1)])
        return actions

    def blue_policy(self, num: int, noise_a_, noise_b_):
        """蓝方策略：0-随机，1-追踪最近敌人，2-不动"""
        actions = {}
        match num:
            case 0:  # 随机策略
                for t in self.blue_tanks:
                    actions[t.uid] = [np.random.uniform(-1, 1) * 2 * math.pi]
                self.assign_actions(actions)
                # 结算伤害
                for t in self.blue_tanks:
                    if t.alive is False:
                        continue
                    for b in t.consumed_bullets:
                        for tt in self.red_tanks:
                            if tt.alive is False:
                                continue
                            distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                            if distance < b.radius:
                                damage = (1 - distance**2 / (b.radius**2)) * b.damage
                                damage = min(damage, tt.hp)
                                tt.hp -= damage
                                tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                                if distance < (b.radius / 2):
                                    tt.fuel -= min(10 // self.step_size, tt.fuel)
                                    tt.fuel = max(0, tt.fuel)

            case 1:  # 智能策略
                for t in self.blue_tanks:
                    red_xy = [[t.x, t.y] for t in self.red_tanks if t.alive is True]
                    if len(red_xy) == 0:
                        continue
                    dis = [math.sqrt((t.x - red_xy[i][0])**2 + (t.y - red_xy[i][1])**2) for i in range(len(red_xy))]
                    min_dis = min(dis)
                    min_index = dis.index(min_dis)
                    angle = math.atan2(red_xy[min_index][1] - t.y, red_xy[min_index][0] - t.x)
                    if angle < 0:
                        angle += 2 * math.pi

                    attack_angle = angle
                    back_angle = angle + math.pi
                    if back_angle > 2 * math.pi:
                        back_angle -= 2 * math.pi

                    noise_a = np.random.normal(0, noise_a_)
                    noise_b = np.random.normal(0, noise_b_)
                    angle = np.clip(angle + noise_a, 0, 2*math.pi)
                    back_angle = np.clip(back_angle + noise_b, 0, 2*math.pi)
                    attack_angle = np.clip(attack_angle + noise_b, 0, 2*math.pi)

                    if 10 < min_dis < 20:
                        actions[t.uid] = [-attack_angle]
                    elif 20 <= min_dis:
                        actions[t.uid] = [angle]
                    elif min_dis < 10:
                        actions[t.uid] = [back_angle]

                self.assign_actions(actions)
                # 结算伤害
                for t in self.blue_tanks:
                    if t.alive is False:
                        continue
                    for b in t.consumed_bullets:
                        for tt in self.tanks:
                            if tt.alive is False:
                                continue
                            distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                            if distance < b.radius:
                                damage = (1 - distance**2 / (b.radius**2)) * b.damage
                                tt.hp -= min(damage, tt.hp)
                                tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                                if distance < (b.radius / 2):
                                    tt.fuel -= min(10 // self.step_size, tt.fuel)
                                    tt.fuel = max(0, tt.fuel)

            case 2:  # 不动策略
                pass

            case 3:  # 随机选择策略
                num = np.random.randint(3)
                self.blue_policy(num, noise_a_, noise_b_)

    def assign_actions(self, actions: dict[str, Action]) -> None:
        """按照uid匹配并执行动作"""
        for uid, a in actions.items():
            flag = False
            for t in self.tanks:
                if uid == t.uid:
                    if t.alive is False:
                        continue
                    if a[0] > 0:
                        t.move(a[0])
                        flag = True
                    elif a[0] < 0:
                        t.shoot(abs(a[0]))
                        flag = True
                    elif a[0] == 0:
                        t.speed = 0
                        flag = True
                if flag is True:
                    break

    def get_reward_t(self):
        """扫描当前环境，使伤害生效，并提取奖励"""
        reward = {t.uid: 0.0 for t in self.red_tanks}

        # 碰壁惩罚
        for t in self.red_tanks:
            if t.x == 0 or t.x == config.map.width or t.y == 0 or t.y == config.map.height:
                reward[t.uid] -= 0.1 * self.step_size

        # 红方坦克的奖励和结算伤害
        for t in self.red_tanks:
            if t.alive is False:
                continue
            for b in t.consumed_bullets:
                for tt in self.tanks:
                    if tt.alive is False:
                        continue
                    distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                    if distance < b.radius:
                        damage = (1 - distance**2 / (b.radius**2)) * b.damage
                        damage = min(damage, tt.hp)
                        tt.hp -= damage
                        tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                        if tt.team == "Blue":
                            reward[t.uid] += damage
                        elif tt.team == "Red":
                            reward[t.uid] -= damage
                        if tt.alive is False:
                            if tt.team == "Blue":
                                reward[t.uid] += 3
                            elif tt.team == "Red":
                                reward[t.uid] -= 3
                        if distance < (b.radius / 2):
                            tt.fuel -= min(10 // self.step_size, tt.fuel)
                            tt.fuel = max(0, tt.fuel)
                            if tt.team == "Blue":
                                reward[t.uid] += 1
                            elif tt.team == "Red":
                                reward[t.uid] -= 1

        for t in self.tanks:
            t.consumed_bullets.clear()

        return reward

    def get_reward(self):
        return self.get_reward_t()

    def get_obs(self) -> Observation:
        """获取环境观测信息"""
        blue_t_xy = [np.array([t.x, t.y], dtype=np.float32) for t in self.blue_tanks]

        obs = {}
        for t in self.red_tanks:
            obs.update({
                t.uid: np.array([
                    t.x, t.y,
                    t.hp,
                    t.speed,
                    len(t.bullets),
                    t.fuel,
                    blue_t_xy[0][0], blue_t_xy[0][1],
                    blue_t_xy[1][0], blue_t_xy[1][1],
                    blue_t_xy[2][0], blue_t_xy[2][1],
                ], dtype=np.float32)
            })
        return obs

    @property
    def terminated(self) -> bool:
        """判断对战是否结束"""
        tem = any(self.red_team.alives) is False or any(self.blue_team.alives) is False
        terminated = {}
        for t in self.agents:
            terminated[t] = tem
        return terminated

    @property
    def truncated(self) -> bool:
        """判断对战是否超出最大回合长度"""
        tr1 = bool(self.step_ >= config.max_steps)
        tr2 = [t.alive is False or len(t.bullets) == 0 for t in self.tanks]
        tr = tr1 or all(tr2)
        truncated = {}
        for t in self.agents:
            truncated[t] = tr
        return truncated

    def get_info(self) -> Info:
        """获取额外信息"""
        infos = {'win1': False, 'win2': False, 'lose1': False, 'lose2': False, 'win': False, 'lose': False}
        tem = any(self.terminated.values())
        tru = any(self.truncated.values())
        done = tem or tru
        if done:
            if tem:
                if any(self.blue_team.alives) is False:
                    infos['win1'] = True
                if any(self.red_team.alives) is False:
                    infos['lose1'] = True
            elif tru:
                hp_red = sum([t.hp for t in self.red_tanks])
                hp_blue = sum([t.hp for t in self.blue_tanks])
                if hp_red > hp_blue:
                    infos['win2'] = True
                if hp_red <= hp_blue:
                    infos['lose2'] = True
            if infos['win1'] or infos['win2']:
                infos['win'] = True
            elif infos['lose1'] or infos['lose2']:
                infos['lose'] = True

        for t in self.tanks:
            infos[t.uid] = t.hp
        for t in self.tanks:
            infos[t.uid + '_xy'] = [int(t.x), int(t.y)]
        for t in self.tanks:
            infos[t.uid + '_fuel'] = t.fuel

        return infos

    def step(self, actions: dict[str, Action]) -> tuple[Observation, Reward, bool, bool, Info]:
        """环境迭代一个时间步"""
        self.step_ += 1
        num = np.random.randint(2)
        if num == 0:
            self.blue_policy(self.policy, self.noise_a, self.noise_b)
            self.assign_actions(actions)
        elif num == 1:
            self.assign_actions(actions)
            self.blue_policy(self.policy, self.noise_a, self.noise_b)

        r = self.get_reward()
        obs = self.get_obs()
        info = self.get_info()

        return obs, r, self.terminated, self.truncated, info

    def reset(self, policy_num=1, noise_a=0.1*2*math.pi, noise_b=0.12*2*math.pi) -> tuple[Observation, Info]:
        """重置环境"""
        for k in Tank.id_counter:
            Tank.id_counter[k] = 0
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0

        self.policy = policy_num
        self.noise_a = noise_a
        self.noise_b = noise_b

        info = {}
        return self.get_obs(), info

    def action_space(self, agent_id) -> spaces.Space:
        action = self.sample()
        return spaces.Box(-1, 1, shape=(len(action[agent_id]),), dtype=np.float32)

    def observation_space(self, agent_id) -> spaces.Space:
        obs = self.get_obs()
        return spaces.Box(0, 1, shape=(len(obs[agent_id]),), dtype=np.float32)

    def Normalization(self, obs):
        """归一化观测值"""
        for k in obs.keys():
            if 'tank' in k:
                obs[k][0] = (obs[k][0] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                obs[k][1] = (obs[k][1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])
                obs[k][2] = (obs[k][2] - config.observation_space["hp_t"][0]) / (config.observation_space["hp_t"][1] - config.observation_space["hp_t"][0])
                obs[k][3] = (obs[k][3] - config.observation_space["speed_t"][0]) / (config.observation_space["speed_t"][1] - config.observation_space["speed_t"][0])
                obs[k][4] = (obs[k][4] - config.observation_space["bullet_t"][0]) / (config.observation_space["bullet_t"][1] - config.observation_space["bullet_t"][0])
                obs[k][5] = (obs[k][5] - config.observation_space["fuel"][0]) / (config.observation_space["fuel"][1] - config.observation_space["fuel"][0])

                n = len(obs[k])
                for i in range(6, n, 2):
                    obs[k][i] = (obs[k][i] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                    obs[k][i+1] = (obs[k][i+1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])

        return obs
