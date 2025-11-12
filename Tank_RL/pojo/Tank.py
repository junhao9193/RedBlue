"""坦克类"""
import math
from collections import deque
from typing import Literal

from pathlib import Path
import addict
import toml

# 类型定义
TeamName = Literal["Red", "Blue"]
Number = float | int

# 加载配置
config = addict.Dict(toml.load(Path(__file__).parent.parent / "config" / "tank.toml"))


class Tank:
    """坦克类

    :param x: 坦克的起始横坐标
    :type x: float
    :param y: 坦克的起始纵坐标
    :type y: float
    :param team: 所属队伍（Red或Blue）
    :type team: TeamName
    """

    id_counter: dict[str, int] = {"Red": 0, "Blue": 0}

    def __init__(self, x: float, y: float, team: TeamName) -> None:
        assert team in TeamName.__args__
        self.x = x
        self.y = y
        self._speed = config.tank.speed
        self.speed = 0
        self.team = team
        self.x_max = config.map.width
        self.y_max = config.map.height
        self.hp = config.tank.hp
        self.fuel = config.tank.fuel
        self.visibility_range = config.tank.visibility_range

        # 导入 Bullet 类
        from .Bullet import Bullet

        self.bullets = deque(
            [
                Bullet(config.bullet.damage_radius, self)
                for _ in range(config.tank.bullet_num)
            ],
        )
        self.consumed_bullets = deque[Bullet]()
        self.shoot_distance = config.tank.shoot_distance
        self.uid = f"{team}-tank-{Tank.id_counter[team]}"
        Tank.id_counter[team] += 1

    @staticmethod
    def clip(v: Number, _min: Number, _max: Number) -> Number:
        """将数值限制在上下界之内"""
        if v > _max:
            return _max
        elif v < _min:
            return _min
        else:
            return v

    @property
    def alive(self) -> bool:
        """指示坦克是否存活"""
        return self.hp > 1e-3

    @property
    def movable(self) -> bool:
        """指示坦克是否还能移动"""
        return self.alive and self.fuel > 0

    @property
    def shootable(self) -> bool:
        """指示坦克是否还能射击"""
        return self.alive and len(self.bullets) > 0

    def move(self, angle: float) -> None:
        r"""指定角度移动

        :param angle: 朝哪个方向移动（弧度制 :math:`0\le angle\le 2\pi`）
        """
        assert 0 <= angle <= 2 * math.pi
        if abs(angle - 2 * math.pi) < 1e-3:
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        if self.movable is False:
            self.speed = 0
            return
        self.speed = self._speed
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        delta_x = self.speed * cos
        delta_y = self.speed * sin

        self.x = Tank.clip(self.x + delta_x, 0, self.x_max)
        self.y = Tank.clip(self.y + delta_y, 0, self.y_max)

        for b in self.bullets:
            b.x = self.x
            b.y = self.y
        self.fuel -= 1
        self.fuel = max(0, self.fuel)

    def shoot(self, angle: float) -> None:
        r"""朝指定角度射击

        :param angle: 朝哪个方向射击（弧度制 :math:`0\le angle\le 2\pi`）
        子弹会移动到 shoot_distance 距离处
        """
        assert 0 <= angle <= 2 * math.pi
        if self.shootable is False:
            return
        if abs(angle - 2 * math.pi) < 1e-3:
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        self.speed = 0
        b = self.bullets.pop()
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        b.x += self.shoot_distance * cos
        b.y += self.shoot_distance * sin
        b.x = Tank.clip(b.x, 0, self.x_max)
        b.y = Tank.clip(b.y, 0, self.y_max)
        self.consumed_bullets.append(b)
