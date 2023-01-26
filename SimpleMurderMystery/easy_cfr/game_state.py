from __future__ import annotations

import random
from abc import ABC, abstractmethod

from enum import Enum, IntEnum
from typing import List, Tuple


class Player(IntEnum):
    TERMINAL = -2
    CHANCE = -1
    P_ONE = 0
    P_TWO = 1


class PlayerInterface(ABC):
    @abstractmethod
    def get_action(self, state: GameState) -> int:
        pass

    @abstractmethod
    def get_inf_set_action(self, state: GameState) -> int:
        pass


class GameState(ABC):

    def is_terminal(self) -> bool:
        pass

    def current_player(self) -> int:
        pass

    def n_actions(self) -> int:
        pass

    def max_actions(self) -> int:
        pass

    def actions(self) -> List[int]:
        pass

    def act(self, action: int) -> None:
        pass

    def child(self, action: int) -> GameState:
        cp = self.copy_state()
        cp.act(action)
        return cp

    def chance_outcomes(self) -> List[Tuple[GameState, float]]:
        assert self.current_player() == Player.CHANCE
        p = 1 / len(self.actions())
        return [(self.child(action), p) for action in self.actions()]

    def chance_action_probs(self) -> List[Tuple[int, float]]:
        assert self.current_player() == Player.CHANCE
        p = 1 / len(self.actions())
        return [(action, p) for action in self.actions()]

    def take_uniform_random_action(self) -> GameState:
        assert self.current_player() == Player.CHANCE
        action = random.choice(self.actions())
        return self.child(action)

    def get_uniform_random_action(self) -> int:
        return random.choice(self.actions())

    def children(self) -> List[GameState]:
        if self.is_terminal():
            return []
        else:
            return [self.child(action) for action in self.actions()]

    def returns(self) -> List[float]:
        pass

    def copy_state(self) -> GameState:
        pass

    def action_to_string(self, action) -> str:
        pass

    def information_set(self) -> str:
        pass

