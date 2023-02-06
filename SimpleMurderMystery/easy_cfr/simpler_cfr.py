from __future__ import annotations

import logging
import random
from functools import partial
from typing import List, Optional, Dict, Tuple

import numpy as np
import numpy.typing as npt

from easy_cfr.evaluate_policies import print_eval
from easy_cfr.game_and_agent_interfaces import GameModel, Player, PlayerInterface
from easy_cfr.kuhn_poker import KuhnPoker
from easy_cfr.murder_mystery import MurderMysteryParams, MurderGameModel
from easy_cfr.policy_player import MyPolicy

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

info_set_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)

info_set_logger.addHandler(c_handler)


class InfoSetTabularPolicy:
    def __init__(self) -> None:
        self.p_action_dict: Dict[str, npt.NDArray] = dict()
        self.regret_dict: Dict[str, npt.NDArray] = dict()
        self.policy_dict: Dict[str, npt.NDArray] = dict()

    def p_actions(self, key: str, n_actions: int) -> npt.NDArray:
        if key in self.p_action_dict:
            pa = self.p_action_dict[key]
            assert len(pa) == n_actions
            return pa
        else:
            pa = np.ones(n_actions)
            pa /= n_actions
            self.p_action_dict[key] = pa
            self.regret_dict[key] = np.zeros(n_actions)
            self.policy_dict[key] = np.ones(n_actions) / n_actions
            return pa

    def regrets(self, key: str) -> npt.NDArray:
        assert key in self.regret_dict
        return self.regret_dict[key]

    @staticmethod
    def normalise(p: npt.NDArray) -> npt.NDArray:
        sum_p = np.sum(p)
        return p / sum_p

    def force_random(self):
        for key, value in self.policy_dict.items():
            n = len(value)
            self.policy_dict[key] = np.ones(n) / n

    def update(self, step: int) -> None:
        for key, regrets in self.regret_dict.items():
            floored_regrets = np.maximum(regrets, 1e-16)
            self.p_action_dict[key] = self.normalise(floored_regrets)
            lr = 1 / (1 + step)
            # todo: need to ensure this works properly
            self.policy_dict[key] *= (1 - lr)
            self.policy_dict[key] += self.p_action_dict[key] * lr
            # print(f"updating for {key=}")
            # print(f"updated: {floored_regrets=}")
            # print(f"updated: {self.policy_dict[key]=}")
            # print(f"updated: {self.p_action_dict[key]=}")


class TabularPolicyPlayer(PlayerInterface):
    def __init__(self, policy: InfoSetTabularPolicy):
        self.policy = policy

    # def make_greedy(self) -> PolicyPlayer:
    #     self.policy = greedify(self.policy)
    #     return self

    def get_action(self, state: GameModel) -> int:
        return self.get_inf_set_action(state)

    def get_inf_set_action(self, state: GameModel) -> int:
        inf_set = state.information_set()
        action_probs = self.get_action_probs(state)
        actions, probs = list(zip(*action_probs))
        choice = random.choices(actions, probs)[0]
        return choice

    def get_action_probs(self, state: GameModel) -> List[Tuple[int, float]]:
        inf_set = state.information_set()
        probs = self.policy.policy_dict[inf_set]
        ap = [(a, p) for a, p in zip(state.actions(), probs)]
        return ap


class FullCFR:
    def __init__(self, game: GameModel, policy: InfoSetTabularPolicy):
        self.game = game
        self.policy = policy
        # hardwire this for now
        self.n_players = 2

    def new_reach(self, so_far: npt.NDArray, player: int, action_prob: float) -> npt.NDArray:
        """Returns new reach probabilities."""
        new = np.array(so_far)
        new[player] *= action_prob
        return new

    def calc_cfr(self, state: GameModel, reach: npt.NDArray) -> npt.NDArray:
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
            inf_set = state.information_set()
            max_actions = state.max_actions()

            # Compute utilities after each action, updating regrets deeper in the tree.
            utility = np.zeros((max_actions, self.n_players))
            # iterate over all the actions, setting the utility for each one
            action_probs = self.policy.p_actions(inf_set, max_actions)
            regrets = self.policy.regrets(inf_set)

            for action in state.actions():
                info_set_logger.info(action_probs)
                prob = action_probs[action]

                # utility is the vector of utilities for each player, given this action
                utility[action] = self.calc_cfr(state.child(action), self.new_reach(reach, player, prob))

            # Compute regrets at this state.
            # the reach vector is the probability that each player played to get here
            # but for the current state we exclude the current player, who is assumed to have played that move intentionally with p=1
            cfr_prob = np.prod(reach[:player]) * np.prod(reach[player + 1:])

            # the value is a vector with the value for each player calculated by mutiplying the utility by the probability of selecting the action
            # value = [u * p for player in range()]
            policy_vec = action_probs
            value = [sum(utility[action][player] * policy_vec[action] for action in state.actions()) for player in
                     range(self.n_players)]
            value = np.array(value)
            info_set_logger.info(f"{value=}")
            for action in state.actions():
                regrets[action] += cfr_prob * (utility[action][player] - value[player])

            # Return the value of this state for all players.
            return value


def run_easy_cfr(state_factory, n_iterations: int = 100) -> InfoSetTabularPolicy:
    policy = InfoSetTabularPolicy()
    initial_state = state_factory()
    full_cfr = FullCFR(initial_state, policy)

    for step in range(n_iterations):
        info_set_logger.info(f"{step=}")
        n_players = 2
        n_players_including_chance = n_players + 1
        values = full_cfr.calc_cfr(initial_state, np.ones(n_players_including_chance))
        policy.update(step)
        # policy.print()
        print(f"{step=}, {values=}")
        # print()

    return policy


def info_set_actions_test():
    istp = InfoSetTabularPolicy()
    pa = istp.p_actions("hello", 3)
    print(pa)


def info_set_cfr_test():
    params = MurderMysteryParams(allow_pass=True, allow_suicide=False, n_people=3, max_turns=6)
    model_factory = partial(MurderGameModel, params)

    model_factory = KuhnPoker

    policy = run_easy_cfr(model_factory, 513)
    print(f"{policy.policy_dict=}")
    print(f"{policy.regret_dict=}")
    print(f"{policy.p_action_dict=}")
    policy_player = TabularPolicyPlayer(policy)
    random_policy = run_easy_cfr(model_factory, 1)
    random_policy.force_random()
    print(random_policy.policy_dict)
    random_player = TabularPolicyPlayer(random_policy)
    print_eval(model_factory, policy_player, random_player)

    for inf_set, ap in policy.policy_dict.items():
        print(f"{inf_set=}, {ap=}")


if __name__ == '__main__':
    info_set_actions_test()
    info_set_logger.info(logging.INFO)
    info_set_cfr_test()
