from open_spiel.python.algorithms import exploitability, cfr

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
import itertools as it

from murderspiel.pyspiel_utilities import get_cfr_policy

game = MurderMysteryVariationsGame()




# cfr_policy = get_cfr_policy(game, 10)
#
# print_policy(cfr_policy)
# print(len(cfr_policy.action_probability_array))

def test_variations():
    # note use of it.product to enumerate all combinations of its arguments
    # but setting allow_pass to true causes a crash
    # there may be a bug in the game,
    # or could be that the trees get too big for the full naive
    # CFR algorithm when passing is allowed - not sure if that's likely
    for allow_pass, allow_suicide in it.product([False, True], [False, True]):
        print(allow_pass, allow_suicide)
        params = MurderMysteryParams(allow_pass, allow_suicide)
        game = MurderMysteryVariationsGame(game_params=params)
        cfr_policy = get_cfr_policy(game, 10)
        # print_policy(cfr_policy)
        print(len(cfr_policy.action_probability_array))


if __name__ == '__main__':
    test_variations()
