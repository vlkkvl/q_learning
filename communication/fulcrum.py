from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

"""
Interface for communication between agent and environment.
"""
class Fulcrum(ABC):
    @abstractmethod
    def execute_action(self, action: int) -> Tuple[np.ndarray, bool, bool]:
        """
        Executes one action in the environment.
        The actions are 0 - left, 1 - right, 2 - up, 3 - down

        :return: (next_state, done, info)
        - next_state: observable state (left_free, right_free, front_free, rule)
        - moved: was agent able to move (no wall)?
        - done: goal reached?
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Returns current observable state of the agent as np.ndarray.
        e.g. [0, 1, 0, 1, 0, 0, 0, 1], where
            [:4] - available actions, where 1 - action free, 0 - action impossible (wall)
            [4:] - one-hot of observed sign
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets environment:
        - moves player to start position
        - sets goal as not reached
        - resets moves to 0
        """
        pass