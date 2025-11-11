"""无人机类"""
import math
from collections import deque
from typing import Literal

# 导入模块
from pathlib import Path
import addict
import toml

# 类型定义
TeamName = Literal["Red", "Blue"]
Number = float | int

# 加载配置
config = addict.Dict(toml.load(Path(__file__).parent.parent / "config" / "uav.toml"))


class Uav:
    """无人机类

    :param x: 无人机x坐标
    :type x: float
    :param y: 无人机y坐标
    :type y: float
    """

    id_counter: dict[str, int] = {"Red": 0, "Blue": 0}

    def __init__(self, x: float, y: float, team: TeamName) -> None:
        assert team in TeamName.__args__
        self.x = x
        self.y = y
        self._speed = config.uav.speed
        self.speed = 0
        self.team = team
        self.x_max = config.map.width
        self.y_max = config.map.height
        self.hp = config.uav.hp
        self.fuel = config.uav.fuel
        
        # 导入 Missile 类
        from .Missile import Missile

        self.missiles = deque(
            [
                Missile(config.missile.damage_radius, self)
                for _ in range(config.uav.missile_num)
            ],
        )
        self.consumed_missiles = deque[Missile]()
        self.shoot_distance = config.uav.shoot_distance
        self.uid = f"{team}-uav-{Uav.id_counter[team]}"
        Uav.id_counter[team] += 1

         # 添加射频单元
        from .RFUnit import RFUnit
        self.rf_unit = RFUnit(self)

    @staticmethod
    def clip(v: Number, _min: Number, _max: Number) -> Number:
        """裁剪函数，限制值在最小值和最大值之间"""
        if v > _max:
            return _max
        elif v < _min:
            return _min
        else:
            return v

    @property
    def alive(self) -> bool:
        """是否存活"""
        return self.hp > 1e-3  # 1e-3 是阈值，小于此值视为死亡

    @property
    def movable(self) -> bool:
        """是否可移动"""
        return self.alive and self.fuel > 0

    @property
    def shootable(self) -> bool:
        """是否可射击"""
        return self.alive and len(self.missiles) > 0

    def move(self, angle: float) -> None:
        r"""移动函数

        :param angle: 移动角度，范围 :math:`0\le angle\le 2\pi`
        """
        assert 0 <= angle <= 2 * math.pi  # + 1e-3
        if abs(angle - 2 * math.pi) < 1e-3:  # 如果角度接近 2π，则设为 2π
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        if self.movable is False:
            self.speed = 0
            return
        self.speed = self._speed
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:  # 避免浮点数精度问题
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        delta_x = self.speed * cos
        delta_y = self.speed * sin

        self.x = Uav.clip(self.x + delta_x, 0, self.x_max)
        self.y = Uav.clip(self.y + delta_y, 0, self.y_max)

        for b in self.missiles:
            b.x = self.x
            b.y = self.y
        # dis = math.sqrt(delta_x ** 2 + delta_y ** 2)
        self.fuel -= 1  ### 应该根据 distance 计算
        self.fuel = max(0, self.fuel)

    def shoot(self, angle: float) -> None:
        r"""射击函数
        :param angle: 射击角度，范围 :math:`0\le angle\le 2\pi`
        子弹会移动到 shoot_distance 距离处
        """
        assert 0 <= angle <= 2 * math.pi  # + 1e-3
        if self.shootable is False:
            return
        if abs(angle - 2 * math.pi) < 1e-3:  # 如果角度接近 2π，则设为 2π
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        self.speed = 0
        b = self.missiles.pop()
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        b.x += self.shoot_distance *  cos
        b.y += self.shoot_distance * sin
        b.x = Uav.clip(b.x, 0, self.x_max)
        b.y = Uav.clip(b.y, 0, self.y_max)
        # print(self.uid,b.x,b.y)
        self.consumed_missiles.append(b)
