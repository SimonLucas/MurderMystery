from __future__ import annotations

import logging
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import numpy.typing as npt

from easy_cfr.game_and_agent_interfaces import GameModel, Player, PlayerInterface

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')


# cfr_logger = logging.getLogger(__name__)
#
# c_handler = logging.StreamHandler()
# c_handler.setLevel(logging.DEBUG)
# # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# # c_handler.setFormatter(c_format)
#
# cfr_logger.addHandler(c_handler)


class MyPolicy:
    def __init__(self, info_sets: Dict[str, int], n_players: int = 2, n_actions: int = 2) -> None:
        self.info_sets = info_sets
        self.n_players = n_players
        self.n_actions = n_actions
        self.policy = self.normalise(np.ones((len(info_sets), n_actions)))
        self.curr_policy = self.policy.copy()
        self.regrets = np.zeros_like(self.policy)

    @staticmethod
    def normalise(p: npt.NDArray) -> npt.NDArray:
        sum_p = np.sum(p, axis=1, keepdims=True)
        return p / sum_p

    def index(self, key: str) -> int:
        return self.info_sets[key]

    def print(self) -> None:
        for k, v in self.info_sets.items():
            print(f"{k:6} p={self.policy[v]}")
        # print(f"{self.policy=}")
        # print(f"{self.curr_policy=}")
        # print(f"{self.regrets=}")

    def update(self, step: int) -> None:
        floored_regrets = np.maximum(self.regrets, 1e-16)
        self.curr_policy = self.normalise(floored_regrets)
        lr = 1 / (1 + step)
        self.policy *= (1 - lr)
        self.policy += self.curr_policy * lr


def greedify(prob_array: npt.NDArray, n_actions: int = 2) -> npt.NDArray:
    greedy_policy = (np.eye(n_actions)[np.argmax(prob_array, axis=1)])
    return greedy_policy


class PolicyPlayer(PlayerInterface):
    def __init__(self, info_set_index: Dict[str, int], policy: npt.NDArray):
        self.info_set_index = info_set_index
        self.policy = policy

    def make_greedy(self) -> PolicyPlayer:
        self.policy = greedify(self.policy)
        return self

    def get_action(self, state: GameModel) -> int:
        return self.get_inf_set_action(state)

    def get_inf_set_action(self, state: GameModel) -> int:
        inf_set = state.information_set()
        index = self.info_set_index[inf_set]
        probs = self.policy[index]
        choice = random.choices(state.actions(), probs)[0]
        history = str(state)
        # confirms this is behaving as expected
        # print(f"{inf_set=}, {index=}, {choice=}, {probs=}, {history=}")
        return choice

    def get_action_probs(self, state: GameModel) -> List[Tuple[int, float]]:
        inf_set = state.information_set()
        index = self.info_set_index[inf_set]
        probs = self.policy[index]
        ap = [(a, p) for a, p in zip(state.actions(), probs)]
        return ap
