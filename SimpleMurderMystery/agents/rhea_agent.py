from __future__ import annotations

# todo: implement a clean way to interface this to the game
# todo implement a do_nothing action


"""
Compared to standard RHEA some differences:
  - Partial Observability
  - Asymmetry for Killer and Detective turns
   
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import random


class Action:
    payload: Optional[object]


class ForwardModel(ABC):
    @abstractmethod
    def score(self) -> float:
        pass

    @abstractmethod
    def is_terminal(self) -> float:
        pass

    @abstractmethod
    def act(self, action: Action) -> float:
        pass

    @abstractmethod
    def get_actions(self, action: Action) -> List[Action]:
        pass

    @abstractmethod
    def copy_model(self) -> ForwardModel:
        pass


class PlayerInterface:
    @abstractmethod
    def get_action(self, model: ForwardModel) -> Action:
        pass


class ActionRHEA(PlayerInterface):
    """
    One of the simplest Statistical Forward Planning algorithms, though performs well across
    a wide range of tasks.
    """

    def __init__(self, l: int = 10, n: int = 20, p_mut: float = 0.2, use_buffer: bool = True, noise_level: float = 0.1):
        self.l: int = l
        self.n: int = n
        self.p_mut: float = p_mut
        self.use_buffer: bool = use_buffer
        self.noise_level: float = noise_level
        self.current: List[float] = []

    def get_action_tuple(self, model: ForwardModel, action_float: float):
        actions = model.get_actions()
        action_index = int(action_float * len(actions))
        return actions[action_index]

    def random_action_sequence(self) -> List[float]:
        return [random.random() for _ in range(self.l)]

    def mutate_sequence(self, seq: List[int]) -> List[int]:
        return [random.random() if random.random() < self.p_mut else x for x in seq]

    def score(self, state: ForwardModel, seq: List[float]) -> float:
        # state_tracker = state_tracker.copy_state()
        # s = state_tracker.copy_state()
        for action_float in seq:
            if state.is_terminal():
                return state.score()
            action = self.get_action_tuple(state, action_float)
            state.act(action)
        return state.score() + random.random() * self.noise_level

    def get_action(self, state: ForwardModel) -> Action:
        if self.use_buffer:
            self.current = self.current or self.random_action_sequence()
        # expected_score is only used for logging
        expected_score: [Union[None, float]] = None
        for i in range(self.n):
            mutated_copy = self.mutate_sequence(self.current)
            current_scored = (self.current, self.score(state.copy_model(), self.current))
            mutated_scored = (mutated_copy, self.score(state.copy_model(), mutated_copy))
            if mutated_scored[1] >= current_scored[1]:
                # print(f"{i=}, {current_scored[1]=}")
                # print(f"{i=}, {mutated_scored[1]=}\n")
                self.current = mutated_scored[0]
                expected_score = mutated_scored[1]
        selected_action_float = self.current[0]
        self.current = self.current[1:]
        self.current.append(random.random())
        print(f"{expected_score=}")
        selected_action = self.get_action_tuple(state, selected_action_float)
        return selected_action

    def __str__(self):
        return "Action-RHEA"
