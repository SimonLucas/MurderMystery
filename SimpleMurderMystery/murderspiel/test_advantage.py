from typing import List
from pyspiel import TabularPolicy

from murderspiel.pyspiel_utilities import get_cfr_policy, total_advantage, print_policy, advantage_as_first_player
from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
from open_spiel.python import policy as policy_lib

from murderspiel.pyspiel_utilities import policy_as_list, sample

if __name__ == '__main__':
    params = MurderMysteryParams(allow_pass=True, allow_suicide=True)
    game = MurderMysteryVariationsGame(game_params=params)
    # game = MurderMysteryVariationsGame()
    # create a uniform random policy
    print(game.game_params)
    uniform_random_policy = policy_lib.TabularPolicy(game)
    cfr_policy = get_cfr_policy(game, 2)
    # cfr_policy = uniform_random_policy
    result = total_advantage(game, cfr_policy, uniform_random_policy)
    print_policy(cfr_policy)
    print(f"Advantage = {result}")
