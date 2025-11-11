# UAV_RL/pojo/RFUnit.py
"""射频单元 - 双信道电磁对抗"""
from pathlib import Path
import addict
import toml

# 加载配置
config = addict.Dict(toml.load(Path(__file__).parent.parent / "config" / "uav.toml"))


class RFUnit:
    """射频单元类

    包含通信信道和干扰信道
    """

    def __init__(self, owner):
        self.owner = owner

        # 信道数量（从配置文件读取）
        self.num_channels = config.RFUnit.num_channels
        
        # 当前使用的信道
        self.comm_channel = 0   # 通信信道 [0-9]
        self.jam_channel = 1    # 干扰信道 [0-9]
        
        # 干扰状态（每回合更新）
        self.is_comm_jammed = False     # 通信是否被干扰
        self.jam_success_count = 0       # 本回合成功干扰敌人数量
    
    def set_channels(self, comm_ch: int, jam_ch: int):
        """
        设置信道（由智能体动作决定）
        
        :param comm_ch: 通信信道 [0-9]
        :param jam_ch: 干扰信道 [0-9]
        """
        self.comm_channel = int(comm_ch) % self.num_channels
        self.jam_channel = int(jam_ch) % self.num_channels
    
    def reset_status(self):
        """重置干扰状态（每回合开始调用）"""
        self.is_comm_jammed = False
        self.jam_success_count = 0