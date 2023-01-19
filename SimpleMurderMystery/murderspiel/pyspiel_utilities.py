from typing import List, Tuple

from pyspiel import TabularPolicy
from open_spiel.python.algorithms import exploitability, cfr

import numpy as np

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
import itertools as it


def policy_as_list(policy: TabularPolicy, state: MurderMysteryVariationsGame):
    policy_list = list(enumerate(policy.policy_for_key(state.information_state_string())))
    return policy_list


def print_policy(policy: TabularPolicy) -> None:
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')


def sample(actions_and_probs: List[Tuple[int, float]]) -> int:
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)


def get_cfr_policy(game: MurderMysteryVariationsGame, n: int) -> TabularPolicy:
    cfr_solver = cfr.CFRSolver(game)
    average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
    return average_policy
