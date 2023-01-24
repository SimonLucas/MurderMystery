from __future__ import annotations

import logging
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import numpy.typing as npt

from easy_cfr.game_state import GameState, Player, PlayerInterface
from easy_cfr.policy_player import MyPolicy

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

cfr_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)

cfr_logger.addHandler(c_handler)


class InfSetPolicy:
    def __init__(self, n_actions: int = 2) -> None:
        self.probs: List[float] = [1 / n_actions for _ in range(n_actions)]

    def p(self) -> List[float]:
        return self.probs


class Policy:
    def __init__(self) -> None:
        pass


class FullCFR:
    def __init__(self, game: GameState, policy: MyPolicy):
        self.game = game
        self.policy = policy
        self.n_actions = policy.n_actions
        self.n_inf_sets = len(policy.info_sets)
        self.n_players = policy.n_players

    def new_reach(self, so_far: npt.NDArray, player: int, action_prob: float):
        """Returns new reach probabilities."""
        new = np.array(so_far)
        new[player] *= action_prob
        return new

    def calc_cfr(self, state: GameState, reach: npt.NDArray) -> npt.NDArray:
        """Updates regrets; returns utility for all players."""
        if state.is_terminal():
            return state.returns()
        elif state.current_player() == Player.CHANCE:
            # print(state.chance_action_probs())
            return sum(prob * self.calc_cfr(state.child(action), self.new_reach(reach, Player.CHANCE, prob))
                       for action, prob in state.chance_action_probs())
        else:
            # We are at a player decision point.
            player = state.current_player()
            index = self.policy.index(state.information_set())
            inf_set = state.information_set()
            # Compute utilities after each action, updating regrets deeper in the tree.
            utility = np.zeros((self.n_actions, self.n_players))
            curr_policy = self.policy.curr_policy
            regrets = self.policy.regrets

            # iterate over all the actions, setting the utility for each one
            for action in state.actions():
                # index is the index for this information set, so it's really not needed here if we just looked it up
                # directly from the InformationSet hash value of the state - just return it as a string for now

                # prob is a scalar: the probability this policy will choose this 'action' at infoset 'index'
                cfr_logger.info(curr_policy)
                prob = curr_policy[index][action]

                # the utility is vector of utilities for each player, given this action
                utility[action] = self.calc_cfr(state.child(action), self.new_reach(reach, player, prob))

            # Compute regrets at this state.
            # the reach vector is the probability that each player played to get here
            # but for the current state we exclude the current player, who is assumed to have played that move intentionally with p=1
            cfr_prob = np.prod(reach[:player]) * np.prod(reach[player + 1:])

            # the value is a vector with the value for each player calculated by mutiplying the utility by the probability of selecting the action
            # value = [u * p for player in range()]
            policy_vec = curr_policy[index]
            value = [sum(utility[action][player] * policy_vec[action] for action in state.actions()) for player in
                     range(self.n_players)]
            value = np.array(value)
            # value = np.einsum('ap,a->p', utility, curr_policy[index])
            cfr_logger.info(f"{value=}")
            for action in state.actions():
                regrets[index][action] += cfr_prob * (utility[action][player] - value[player])

            # Return the value of this state for all players.
            return value

