"""3v3无人机电磁对抗环境实现"""
import math
import numpy as np

## 数据类型
from collections import deque
from typing import Literal

## 配置
from pathlib import Path
import addict
import toml

# 读取配置
config = addict.Dict(toml.load(Path(__file__).parent / "config" / "uav.toml"))
config.observation_space = {
    # 共用
    "x": [0, config.map.width],
    "y": [0, config.map.height],

    # UAV
    "hp": [0, config.uav.hp],
    "speed": [0, config.uav.speed],
    "missile": [0, config.uav.missile_num],
    "fuel": [0, config.uav.fuel],
    "channel": [0, config.RFUnit.num_channels - 1],
}

# 导入UAV相关类
import sys
sys.path.append(str(Path(__file__).parent))
from pojo.Uav import Uav
from pojo.Missile import Missile
from pojo.RFUnit import RFUnit

# 类型定义
TeamName = Literal["Red", "Blue"]
Action = np.ndarray
Observation = dict[str, np.ndarray]
Reward = dict[str, float]
Info = dict

__all__ = ["UAVEnv", "config"]


class Team:
    """队伍类实现，管理一个队伍的所有UAV

    :param name: 队伍名称 ("Red" 或 "Blue")
    """

    def __init__(self, name: TeamName) -> None:
        assert name in ["Red", "Blue"]
        self.name = name
        # 红方在地图下方(y=0)，蓝方在地图上方(y=height)
        _y = config.map.height if self.name == "Blue" else 0

        # 创建UAV列表
        self.uavs = [
            Uav(
                config.map.width / 2
                + config.map.width / 5 * ((i + 1) // 2) * (-1) ** i,
                _y,
                self.name,
            )
            for i in range(config.team.get(f"n_{self.name.lower()}_uav"))
        ]

    @property
    def alives(self) -> list[bool]:
        """获取队伍所有UAV的存活状态"""
        return [uav.alive for uav in self.uavs]


class UAVEnv:
    """
    3v3无人机电磁对抗环境

    特点：
    1. 动作空间：3维 (angle, comm_channel, jam_channel)
    2. 电磁对抗：通信被干扰时看不到队友
    3. 干扰成功造成伤害
    4. 切换信道消耗燃油
    """

    def __init__(self, render_mode=False) -> None:
        self.render_mode = render_mode
        self.step_size = config.max_steps / config.max_steps  # 归一化步长
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0  # 记录当前回合数

        # 要进行训练的智能体（红方）
        self.agents = [uav.uid for uav in self.red_team.uavs]
        # 敌方智能体（蓝方）
        self.agents_e = [uav.uid for uav in self.blue_team.uavs]
        self.episode_length = config.max_steps

        # 蓝方信道轮询计数器（用于round-robin channel selection）
        self.blue_comm_channel_cycle = 0
        self.blue_jam_channel_cycle = 0

    # ==================== 环境属性 ====================
    @property
    def num_agents(self) -> int:
        """训练智能体数量"""
        return len(self.agents)

    @property
    def uavs(self) -> list[Uav]:
        """所有的UAV实例"""
        return self.red_team.uavs + self.blue_team.uavs

    @property
    def red_uavs(self) -> list[Uav]:
        """红队的UAV实例"""
        return self.red_team.uavs

    @property
    def blue_uavs(self) -> list[Uav]:
        """蓝队的UAV实例"""
        return self.blue_team.uavs

    @property
    def terminated(self) -> dict[str, bool]:
        """判断对战是否结束（一方全灭）"""
        tem = any(self.red_team.alives) is False or any(self.blue_team.alives) is False
        terminated = {agent: tem for agent in self.agents}
        return terminated

    @property
    def truncated(self) -> dict[str, bool]:
        """判断对战是否超出最大回合或无法继续"""
        tr1 = bool(self.step_ >= config.max_steps)
        # 所有UAV都无法攻击（死亡或无导弹）
        tr2 = [uav.alive is False or len(uav.missiles) == 0 for uav in self.uavs]
        tr = tr1 or all(tr2)
        truncated = {agent: tr for agent in self.agents}
        return truncated

    # ==================== 核心方法 ====================

    def sample(self) -> dict[str, Action]:
        """随机生成动作

        返回格式: {uid: [angle, comm_ch, jam_ch]}
        - angle: 移动角度或射击角度（正=移动，负=射击）[-2π, 2π]
        - comm_ch: 通信信道 [0, num_channels-1]
        - jam_ch: 干扰信道 [0, num_channels-1]
        """
        actions = {}
        for uav in self.red_team.uavs:
            angle = np.random.uniform(-1, 1) * 2 * math.pi
            comm_ch = np.random.randint(0, config.RFUnit.num_channels)
            jam_ch = np.random.randint(0, config.RFUnit.num_channels)
            actions[uav.uid] = np.array([angle, comm_ch, jam_ch])
        return actions

    def blue_policy(self, policy_num: int = 0) -> None:
        """蓝方策略

        0: 随机策略
        1: 简单规则策略（向最近敌人靠近并攻击）
        2: 不动策略
        """
        actions = {}

        if policy_num == 0:  # 随机策略（蓝方信道按轮询选取，但仅在实际切换时推进计数器）
            for uav in self.blue_uavs:
                angle = np.random.uniform(-1, 1) * 2 * math.pi
                # 使用当前轮询指针选择信道（计数器推进放在实际切换处）
                comm_ch = self.blue_comm_channel_cycle % config.RFUnit.num_channels
                jam_ch = self.blue_jam_channel_cycle % config.RFUnit.num_channels
                actions[uav.uid] = np.array([angle, comm_ch, jam_ch])

            self.assign_actions(actions)
            self._resolve_blue_damage()

        elif policy_num == 1:  # 简单规则策略（蓝方信道按轮询选取，但仅在实际切换时推进计数器）
            for uav in self.blue_uavs:
                if not uav.alive:
                    continue

                # 找到最近的活着的敌人
                red_uavs_alive = [u for u in self.red_uavs if u.alive]
                if len(red_uavs_alive) == 0:
                    continue

                distances = [
                    math.sqrt((uav.x - ru.x)**2 + (uav.y - ru.y)**2)
                    for ru in red_uavs_alive
                ]
                min_index = distances.index(min(distances))
                target = red_uavs_alive[min_index]
                min_dis = distances[min_index]

                # 计算朝向目标的角度
                angle = math.atan2(target.y - uav.y, target.x - uav.x)
                if angle < 0:
                    angle += 2 * math.pi

                # 决策：射击或移动
                if 10 < min_dis < 20:  # 在射击范围内
                    # 使用当前轮询指针选择信道（计数器推进放在实际切换处）
                    comm_ch = self.blue_comm_channel_cycle % config.RFUnit.num_channels
                    jam_ch = self.blue_jam_channel_cycle % config.RFUnit.num_channels
                    actions[uav.uid] = np.array([-angle, comm_ch, jam_ch])  # 负数表示射击
                elif min_dis >= 20:  # 太远，靠近
                    # 使用当前轮询指针选择信道（计数器推进放在实际切换处）
                    comm_ch = self.blue_comm_channel_cycle % config.RFUnit.num_channels
                    jam_ch = self.blue_jam_channel_cycle % config.RFUnit.num_channels
                    actions[uav.uid] = np.array([angle, comm_ch, jam_ch])  # 正数表示移动
                else:  # 太近，后退
                    back_angle = (angle + math.pi) % (2 * math.pi)
                    # 使用当前轮询指针选择信道（计数器推进放在实际切换处）
                    comm_ch = self.blue_comm_channel_cycle % config.RFUnit.num_channels
                    jam_ch = self.blue_jam_channel_cycle % config.RFUnit.num_channels
                    actions[uav.uid] = np.array([back_angle, comm_ch, jam_ch])

            self.assign_actions(actions)
            self._resolve_blue_damage()

        elif policy_num == 2:  # 不动策略
            pass

    def assign_actions(self, actions: dict[str, Action]) -> None:
        """按照uid匹配并执行动作

        :param actions: {uid: [angle, comm_ch, jam_ch]}

        规则：
        - angle < 0: 射击状态，可以更改信道并开火
        - angle == 0: 停止状态，只更改信道，不开火
        - angle > 0: 移动状态，不更改信道
        """
        for uid, action in actions.items():
            uav = None
            for u in self.uavs:
                if u.uid == uid:
                    uav = u
                    break

            if uav is None or not uav.alive:
                continue

            angle, comm_ch, jam_ch = action[0], int(action[1]), int(action[2])

            # 移动/停下/射击
            if angle > 0:  # 移动（不改变信道）
                uav.move(angle)
            elif angle < 0:  # 射击（先更改信道，再开火）
                uav.rf_unit.set_channels(comm_ch, jam_ch)
                # 蓝方信道轮询计数器仅在实际切换时推进
                if uav.team == "Blue":
                    self.blue_comm_channel_cycle = (self.blue_comm_channel_cycle + 1)
                    self.blue_jam_channel_cycle = (self.blue_jam_channel_cycle + 1)
                uav.shoot(abs(angle))
            else:  # angle == 0, 停止（只更改信道，不开火）
                uav.speed = 0
                uav.rf_unit.set_channels(comm_ch, jam_ch)
                if uav.team == "Blue":
                    self.blue_comm_channel_cycle = (self.blue_comm_channel_cycle + 1)
                    self.blue_jam_channel_cycle = (self.blue_jam_channel_cycle + 1)

    def resolve_em_warfare(self) -> None:
        """解析电磁对抗

        规则（简化版）：
        1. comm_channel: 防御信道（我的防御频率）
        2. jam_channel: 攻击信道（攻击敌人的频率）
        3. 干扰成功条件：
           - attacker.jam_channel == target.comm_channel（信道匹配）
           - distance(attacker, target) <= visibility_range（在范围内）
        4. 干扰效果：
           - 目标被干扰：扣除HP
           - 攻击者：成功干扰计数+1
        """
        # 1. 重置所有干扰状态
        for uav in self.uavs:
            uav.rf_unit.reset_status()

        # 2. 检查所有UAV对之间的干扰关系
        for attacker in self.uavs:
            if not attacker.alive:
                continue

            for target in self.uavs:
                if not target.alive or attacker.uid == target.uid:
                    continue

                # 计算距离
                distance = math.sqrt(
                    (attacker.x - target.x)**2 + (attacker.y - target.y)**2
                )

                # 干扰条件1：信道匹配
                # 干扰条件2：在可见范围内
                
                if (attacker.rf_unit.jam_channel == target.rf_unit.comm_channel and
                    distance <= config.uav.visibility_range):
                    target.rf_unit.is_comm_jammed = True
                    
                    # 区分敌人和友军
                    if target.team != attacker.team:
                        attacker.rf_unit.jam_success_enemy_count += 1  # 干扰敌人
                    else:
                        attacker.rf_unit.jam_success_ally_count += 1   # 误伤友军

        # 3. 应用干扰伤害
        for uav in self.uavs:
            if uav.rf_unit.is_comm_jammed:
                damage = config.RFUnit.jam_damage
                uav.hp -= damage
                uav.hp = max(0, uav.hp)

    def _resolve_blue_damage(self) -> None:
        """解析蓝方的导弹伤害（不给奖励，只扣HP）"""
        for uav in self.blue_uavs:
            if not uav.alive:
                continue

            for missile in uav.consumed_missiles:
                for target in self.uavs:
                    if not target.alive:
                        continue

                    distance = math.sqrt(
                        (missile.x - target.x)**2 + (missile.y - target.y)**2
                    )

                    if distance < missile.radius:
                        # 计算伤害（距离越近伤害越大）
                        damage = (1 - distance**2 / missile.radius**2) * missile.damage
                        damage = min(damage, target.hp)
                        target.hp -= damage
                        target.hp = max(0, target.hp)

        # 清除蓝方已消耗的导弹
        for uav in self.blue_uavs:
            uav.consumed_missiles.clear()

    def get_reward(self) -> Reward:
        """计算奖励

        奖励包括：
        1. 导弹伤害奖励/惩罚
        2. 击杀奖励/惩罚
        3. 燃油减少奖励/惩罚
        4. 电磁对抗奖励/惩罚
        5. 碰壁惩罚
        """
        reward = {uav.uid: 0.0 for uav in self.red_uavs}

        # 1. 碰壁惩罚
        for uav in self.red_uavs:
            if (uav.x == 0 or uav.x == config.map.width or
                uav.y == 0 or uav.y == config.map.height):
                reward[uav.uid] += config.reward.wall_collision_penalty * self.step_size

        # 2. 导弹伤害奖励和击杀奖励
        for uav in self.red_uavs:
            if not uav.alive:
                continue

            for missile in uav.consumed_missiles:
                for target in self.uavs:
                    if not target.alive:
                        continue

                    distance = math.sqrt(
                        (missile.x - target.x)**2 + (missile.y - target.y)**2
                    )

                    if distance < missile.radius:
                        # 计算伤害
                        damage = (1 - distance**2 / missile.radius**2) * missile.damage
                        damage = min(damage, target.hp)
                        was_alive = target.hp > 0
                        target.hp -= damage
                        target.hp = max(0, target.hp)

                        # 伤害奖励
                        if target.team == "Blue":
                            reward[uav.uid] += damage * config.reward.damage_reward_scale
                        elif target.team == "Red":
                            reward[uav.uid] += damage * config.reward.damage_penalty_scale

                        # 击杀奖励
                        if was_alive and target.hp <= 0:
                            if target.team == "Blue":
                                reward[uav.uid] += config.reward.kill_reward
                            elif target.team == "Red":
                                reward[uav.uid] += config.reward.kill_penalty

                        # 燃油减少奖励（如果击中核心区域）
                        if distance < missile.radius / 2:
                            fuel_loss = min(10 / self.step_size, target.fuel)
                            target.fuel -= fuel_loss
                            target.fuel = max(0, target.fuel)

                            if target.team == "Blue":
                                reward[uav.uid] += config.reward.fuel_reduction_reward
                            elif target.team == "Red":
                                reward[uav.uid] += config.reward.fuel_reduction_penalty

        # 清除红方已消耗的导弹
        for uav in self.red_uavs:
            uav.consumed_missiles.clear()

        # 3. 电磁对抗奖励
        for uav in self.red_uavs:
            # 成功干扰敌人的奖励
            reward[uav.uid] += uav.rf_unit.jam_success_enemy_count * config.reward.jam_reward

            # 误伤友军的惩罚
            reward[uav.uid] += uav.rf_unit.jam_success_ally_count * config.reward.jam_ally_penalty

            # 被干扰的惩罚
            if uav.rf_unit.is_comm_jammed:
                reward[uav.uid] += config.reward.jammed_penalty

        return reward

    def get_obs(self) -> Observation:
        """获取环境观测信息（简化版）

        观测维度：
        - 自己信息：x, y, hp, speed, missiles, fuel, comm_ch, jam_ch (8维)
        - 敌人信息：x, y（只有位置）(2维 x 3)

        总计：8 + 6 = 14维

        设计理念：
        - 不观测队友（去中心化决策）
        - 信道作为攻防手段，不作为通信手段
        - comm_channel = 防御信道，jam_channel = 攻击信道
        """
        obs = {}

        for uav in self.red_uavs:
            # 自己的信息
            my_obs = [
                uav.x, uav.y, uav.hp, uav.speed,
                len(uav.missiles), uav.fuel,
                uav.rf_unit.comm_channel, uav.rf_unit.jam_channel
            ]

            # 敌人信息（只有位置，视觉可见）
            for enemy in self.blue_uavs:
                my_obs.extend([enemy.x, enemy.y])

            obs[uav.uid] = np.array(my_obs, dtype=np.float32)

        return obs

    def get_info(self) -> Info:
        """获取额外信息

        包括：
        - 胜负信息
        - 每个UAV的状态
        """
        infos = {
            'win1': False, 'win2': False,
            'lose1': False, 'lose2': False,
            'win': False, 'lose': False
        }

        tem = any(self.terminated.values())
        tru = any(self.truncated.values())
        done = tem or tru

        if done:
            if tem:  # 因为一方全灭结束
                if any(self.blue_team.alives) is False:
                    infos['win1'] = True
                if any(self.red_team.alives) is False:
                    infos['lose1'] = True
            elif tru:  # 因为截断结束（比较HP）
                hp_red = sum([uav.hp for uav in self.red_uavs])
                hp_blue = sum([uav.hp for uav in self.blue_uavs])
                if hp_red > hp_blue:
                    infos['win2'] = True
                if hp_red <= hp_blue:
                    infos['lose2'] = True

            if infos['win1'] or infos['win2']:
                infos['win'] = True
            elif infos['lose1'] or infos['lose2']:
                infos['lose'] = True

        # 传出每个UAV的状态
        for uav in self.uavs:
            infos[uav.uid] = uav.hp
            infos[uav.uid + '_xy'] = [int(uav.x), int(uav.y)]
            infos[uav.uid + '_fuel'] = uav.fuel
            infos[uav.uid + '_jammed'] = uav.rf_unit.is_comm_jammed

        return infos

    def step(self, actions: dict[str, Action]) -> tuple[Observation, Reward, dict, dict, Info]:
        """环境迭代一个时间步

        :param actions: {uid: [angle, comm_ch, jam_ch]}
        :return: (obs, reward, terminated, truncated, info)
        """
        self.step_ += 1

        # 随机决定红蓝方动作顺序
        num = np.random.randint(2)
        if num == 0:
            self.blue_policy(0)  # 蓝方先动
            self.assign_actions(actions)  # 红方后动
        else:
            self.assign_actions(actions)  # 红方先动
            self.blue_policy(0)  # 蓝方后动

        # 解析电磁对抗（必须在导弹伤害之后）
        self.resolve_em_warfare()

        # 获取奖励、观测、信息
        reward = self.get_reward()
        obs = self.Normalization(self.get_obs())
        info = self.get_info()

        return obs, reward, self.terminated, self.truncated, info

    def reset(self, policy_num: int = 0) -> tuple[Observation, Info]:
        """重置环境

        :param policy_num: 蓝方策略编号
        :return: (obs, info)
        """
        # 重置计数器
        Uav.id_counter = {"Red": 0, "Blue": 0}

        # 重新创建队伍
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0
        self.policy = policy_num

        # 重置蓝方信道轮询计数器
        self.blue_comm_channel_cycle = 0
        self.blue_jam_channel_cycle = 0

        return self.Normalization(self.get_obs()), {}

    def action_space(self, agent_id):
        """返回动作空间

        3维: [angle, comm_ch, jam_ch]
        - angle: [-2π, 2π]
        - comm_ch: [0, num_channels-1]
        - jam_ch: [0, num_channels-1]
        """
        from gymnasium import spaces
        return spaces.Box(
            low=np.array([-2*math.pi, 0, 0]),
            high=np.array([2*math.pi, config.RFUnit.num_channels-1, config.RFUnit.num_channels-1]),
            dtype=np.float32
        )

    def observation_space(self, agent_id):
        """返回观测空间"""
        from gymnasium import spaces
        obs = self.get_obs()
        return spaces.Box(0, 1, shape=(len(obs[agent_id]),), dtype=np.float32)

    def Normalization(self, obs: Observation) -> Observation:
        """归一化观测值到[0,1]范围

        根据 config.observation_space 中定义的范围进行归一化

        观测结构：
        - [0:2]   位置 (x, y)
        - [2]     hp
        - [3]     speed
        - [4]     missiles
        - [5]     fuel
        - [6]     comm_channel
        - [7]     jam_channel
        - [8:14]  敌人位置 (x, y) × 3

        :param obs: 原始观测字典
        :return: 归一化后的观测字典
        """
        normalized_obs = {}

        for k in obs.keys():
            obs_copy = obs[k].copy()

            # 归一化自己的状态
            # x, y 位置
            obs_copy[0] = (obs_copy[0] - config.observation_space["x"][0]) / \
                         (config.observation_space["x"][1] - config.observation_space["x"][0])
            obs_copy[1] = (obs_copy[1] - config.observation_space["y"][0]) / \
                         (config.observation_space["y"][1] - config.observation_space["y"][0])

            # hp
            obs_copy[2] = (obs_copy[2] - config.observation_space["hp"][0]) / \
                         (config.observation_space["hp"][1] - config.observation_space["hp"][0])

            # speed
            obs_copy[3] = (obs_copy[3] - config.observation_space["speed"][0]) / \
                         (config.observation_space["speed"][1] - config.observation_space["speed"][0])

            # missiles
            obs_copy[4] = (obs_copy[4] - config.observation_space["missile"][0]) / \
                         (config.observation_space["missile"][1] - config.observation_space["missile"][0])

            # fuel
            obs_copy[5] = (obs_copy[5] - config.observation_space["fuel"][0]) / \
                         (config.observation_space["fuel"][1] - config.observation_space["fuel"][0])

            # comm_channel
            obs_copy[6] = (obs_copy[6] - config.observation_space["channel"][0]) / \
                         (config.observation_space["channel"][1] - config.observation_space["channel"][0])

            # jam_channel
            obs_copy[7] = (obs_copy[7] - config.observation_space["channel"][0]) / \
                         (config.observation_space["channel"][1] - config.observation_space["channel"][0])

            # 归一化敌人位置（从第8维开始，每2维是一个敌人的x, y）
            n = len(obs_copy)
            for i in range(8, n, 2):
                # x 坐标
                obs_copy[i] = (obs_copy[i] - config.observation_space["x"][0]) / \
                             (config.observation_space["x"][1] - config.observation_space["x"][0])
                # y 坐标
                obs_copy[i+1] = (obs_copy[i+1] - config.observation_space["y"][0]) / \
                               (config.observation_space["y"][1] - config.observation_space["y"][0])

            normalized_obs[k] = obs_copy

        return normalized_obs
