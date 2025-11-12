from typing import Any, Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

Number: TypeAlias = float | int
TeamName: TypeAlias = Literal["Red", "Blue"]
Observation: TypeAlias = dict[str, int]
Action: TypeAlias = list[Number] | NDArray[np.int_]
Reward: TypeAlias = dict[str, int]
Info: TypeAlias = dict[str, Any]
