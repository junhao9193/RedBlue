# UAV_RL/pojo/RFUnit.py
"""射频单元 - 双信道电磁对抗（简化版）"""
from pathlib import Path
import random
import addict
import toml

# 加载配置
config = addict.Dict(toml.load(Path(__file__).parent.parent / "config" / "uav.toml"))


class RFUnit:
    """射频单元类（简化版）

    信道功能定义：
    - comm_channel: 防御信道（我的防御频率）
    - jam_channel: 攻击信道（攻击敌人的频率）

    干扰机制：
    - 如果敌人的 jam_channel == 我的 comm_channel，且距离 <= 25，我被干扰
    - 被干扰效果：扣除HP
    """

    def __init__(self, owner):
        self.owner = owner

        # 信道数量（从配置文件读取）
        self.num_channels = config.RFUnit.num_channels

        # 当前使用的信道（随机初始化，促进探索）
        self.comm_channel = random.randint(0, self.num_channels - 1)   # 防御信道 [0-9]
        self.jam_channel = random.randint(0, self.num_channels - 1)    # 攻击信道 [0-9]

        # 记录上一步的信道（用于判断是否切换）
        self.prev_comm_channel = self.comm_channel
        self.prev_jam_channel = self.jam_channel

        # 干扰状态（每回合更新）
        self.is_comm_jammed = False
        self.jam_success_enemy_count = 0  # 成功干扰敌人数量
        self.jam_success_ally_count = 0   # 误伤友军数量（友伤）

    def set_channels(self, comm_ch: int, jam_ch: int):
        """
        设置信道（由智能体动作决定）

        切换信道会消耗燃油：
        - 切换防御信道：消耗 comm_switch_cost 燃油
        - 切换攻击信道：消耗 jam_switch_cost 燃油
        - 两个都切换：消耗总和

        :param comm_ch: 防御信道 [0-9]
        :param jam_ch: 攻击信道 [0-9]
        """
        new_comm = int(comm_ch) % self.num_channels
        new_jam = int(jam_ch) % self.num_channels

        # 计算切换成本
        fuel_cost = 0
        if new_comm != self.prev_comm_channel:
            fuel_cost += config.RFUnit.comm_switch_cost
        if new_jam != self.prev_jam_channel:
            fuel_cost += config.RFUnit.jam_switch_cost

        # 扣除燃油
        self.owner.fuel -= fuel_cost
        self.owner.fuel = max(0, self.owner.fuel)

        # 更新上一步的信道记录
        self.prev_comm_channel = self.comm_channel
        self.prev_jam_channel = self.jam_channel

        # 更新当前信道
        self.comm_channel = new_comm
        self.jam_channel = new_jam
    
    def reset_status(self):
        self.is_comm_jammed = False
        self.jam_success_enemy_count = 0
        self.jam_success_ally_count = 0