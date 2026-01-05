from enum import Enum
from typing import List, Tuple

class SignType(Enum):
    # one-hot encoding for sign types (add new ones here)
    NO_RULE: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    DEAD_END: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 1)
    AVOID_LEFT: Tuple[int, int, int, int, int] = (0, 0, 0, 1, 0)
    AVOID_RIGHT: Tuple[int, int, int, int, int] = (0, 0, 1, 0, 0)
    AVOID_UP: Tuple[int, int, int, int, int] = (0, 1, 0, 0, 0)
    AVOID_DOWN: Tuple[int, int, int, int, int] = (1, 0, 0, 0, 0)

    @property
    def vector(self) -> Tuple[int, int, int, int, int]:
        return self.value

    def as_list(self) -> List[int]:
        return list(self.value)

    # mapping from vector to SignType
    @classmethod
    def from_vector(cls, vector: List[int] | Tuple[int, int, int, int, int]):
        vec = tuple(vector)
        for member in cls:
            if member.value == vec:
                return member
        return None