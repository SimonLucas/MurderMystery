from typing import List
import numpy as np
import numpy.typing as npt

from open_spiel.python.algorithms import exploitability, cfr
from pyspiel import TabularPolicy

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
import itertools as it
from open_spiel.python import policy as policy_lib


# now play some games against each other
def sample(actions_and_probs: list):
    actions, probs = zip(*actions_and_probs)
    return np.random.choice(actions, p=probs)


def policy_as_list(policy: TabularPolicy, state: MurderMysteryVariationsGame):
    # print(policy)
    # print(state)

    policy_list = list(enumerate(policy.policy_for_key(state.information_state_string())))
    # print(f"{policy_list=}")
    return policy_list


# this should work for any openspiel game, but I've been lazy and put this type for now
def run_game(game: MurderMysteryVariationsGame, players: List[TabularPolicy]) -> List[float]:
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            action_probs = state.chance_outcomes()
        else:
            player_policy = players[state.current_player()]
            action_probs = policy_as_list(player_policy, state)
        action = sample(action_probs)
        print(f"Player {state.current_player()} took action {action} in state: {state}")
        state = state.child(action)
    print(f"Final state: {state}")
    return state.returns()


if __name__ == '__main__':
    params = MurderMysteryParams(allow_pass=True, allow_suicide=False)
    game = MurderMysteryVariationsGame(game_params=params)
    # create a uniform random policy
    print(game.game_params)
    uniform_random_policy = policy_lib.TabularPolicy(game)
    player_policies = [uniform_random_policy, uniform_random_policy]
    result = run_game(game, player_policies)
    print(result)
