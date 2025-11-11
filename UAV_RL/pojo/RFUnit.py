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

        # 干扰状态（每回合更新）
        self.is_comm_jammed = False     # 是否被干扰
        self.jam_success_count = 0       # 本回合成功干扰敌人数量

    def set_channels(self, comm_ch: int, jam_ch: int):
        """
        设置信道（由智能体动作决定）

        :param comm_ch: 防御信道 [0-9]
        :param jam_ch: 攻击信道 [0-9]
        """
        self.comm_channel = int(comm_ch) % self.num_channels
        self.jam_channel = int(jam_ch) % self.num_channels
    
    def reset_status(self):
        """重置干扰状态（每回合开始调用）"""
        self.is_comm_jammed = False
        self.jam_success_count = 0