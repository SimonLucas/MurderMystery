import logging
from typing import Dict
import numpy as np

from easy_cfr.cfr import FullCFR
from easy_cfr.game_state import GameModel
from easy_cfr.policy_player import MyPolicy, PolicyPlayer

policy_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)

policy_logger.addHandler(c_handler)

def get_info_sets(state: GameModel, info_set_index: Dict[str, int]) -> Dict[str, int]:
    ix = len(info_set_index.items())
    key = state.information_set()
    policy_logger.info(f"{key=}")
    if not key in info_set_index.keys() and not (key == ''):
        info_set_index[key] = ix

    for child in state.children():
        get_info_sets(child, info_set_index)
    return info_set_index


def get_info_sets_per_player(state: GameModel, info_set_index: Dict[str, int], player_id: int) -> Dict[str, int]:
    ix = len(info_set_index.items())
    key = state.information_set()
    if not key in info_set_index.keys() and not (key == '') and state.current_player() == player_id:
        info_set_index[key] = ix

    for child in state.children():
        get_info_sets_per_player(child, info_set_index, player_id)
    return info_set_index

class PolicyHelper:
    def get_policy(self, state: GameModel, n_players: int = 2) -> MyPolicy:
        info_sets = get_info_sets(state, {})
        my_policy = MyPolicy(info_sets, n_players, n_actions=5)
        # my_policy.print()
        return my_policy



def run_cfr(state_factory, n_iterations: int = 100) -> MyPolicy:
    policy = PolicyHelper().get_policy(state_factory())
    initial_state = state_factory()
    full_cfr = FullCFR(initial_state, policy)

    for step in range(n_iterations):
        policy_logger.info(f"{step=}")
        values = full_cfr.calc_cfr(initial_state, np.ones(1 + policy.n_players))
        policy.update(step)
        # policy.print()
        # print(f"{step=}, {values=}")
        # print()

    return policy

def get_policy_player(state_factory, n_iterations: int = 5) -> PolicyPlayer:
    policy = run_cfr(state_factory, n_iterations)
    policy_player = PolicyPlayer(policy.info_sets, policy.policy)
    return policy_player


def get_uniform_policy_player(state_factory) -> PolicyPlayer:
    policy = PolicyHelper().get_policy(state_factory())
    policy_player = PolicyPlayer(policy.info_sets, policy.policy)
    return policy_player

