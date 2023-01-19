from typing import List, Tuple

import pyspiel
from pyspiel import TabularPolicy, State, PlayerId, Game
from open_spiel.python.algorithms import exploitability, cfr

import numpy as np

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
import itertools as it


def policy_as_list(policy: TabularPolicy, state: MurderMysteryVariationsGame):
    # print(state.current_player())
    # print(policy.states_per_player)
    # print("information_string:",  state.information_state_string())
    policy_list = list(enumerate(policy.policy_for_key(state.information_state_string())))
    # print("Returning: ", policy_list)
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


def advantage_as_first_player(state: State, player: TabularPolicy, opponent: TabularPolicy, player_role: int) -> float:
    """
    Computes the advantage (expected reward) for player compared to opponent when player only plays as player one
    """
    if state.is_terminal():
        return state.returns()[player_role]
    elif state.current_player() == PlayerId.CHANCE:
        ap = state.chance_outcomes()
    elif state.current_player() == player_role:
        ap = policy_as_list(player, state)
    elif state.current_player() == 1 - player_role:
        ap = policy_as_list(opponent, state)
    else:
        raise Exception("Should not be here")
    return sum(p * advantage_as_first_player(state.child(a), player, opponent, player_role) for a, p in ap)


def total_advantage(game: Game, player: TabularPolicy, opponent: TabularPolicy) -> float:
    """
    Computes the total advantage (expected reward) for player compared to opponent.
    """
    results = (advantage_as_first_player(game.new_initial_state(), player, opponent, player_role=0),
               + advantage_as_first_player(game.new_initial_state(), player, opponent, player_role=1))
    print(f"Results: {results}")
    return sum(results)
