"""导弹类"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Uav import Uav


class Missile:
    r"""导弹类

    :param radius: 导弹半径
    :param owner: 导弹拥有者
    :param damage: 伤害值
    """

    def __init__(self, radius: float, owner: 'Uav', damage: float = 3) -> None:
        self.radius = radius
        self.owner = owner
        self.x = owner.x
        self.y = owner.y
        self.damage = damage
