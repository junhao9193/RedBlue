"""子弹类"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Tank import Tank


class Bullet:
    r"""子弹类

    :param radius: 杀伤半径
    :param owner: 子弹的所有者
    :param damage: 中心伤害
    """

    def __init__(self, radius: float, owner: 'Tank', damage: float = 3) -> None:
        self.radius = radius
        self.owner = owner
        self.x = owner.x
        self.y = owner.y
        self.damage = damage
