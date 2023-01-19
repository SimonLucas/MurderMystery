from typing import List
from pyspiel import TabularPolicy

from murderspiel.pyspiel_utilities import get_cfr_policy
from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
from open_spiel.python import policy as policy_lib

from murderspiel.pyspiel_utilities import policy_as_list, sample


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
    cfr_policy = get_cfr_policy(game, 1)
    player_policies = [uniform_random_policy, cfr_policy]
    result = run_game(game, player_policies)
    print(result)
