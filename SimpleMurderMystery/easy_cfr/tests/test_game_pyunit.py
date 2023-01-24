
# from unittest import main, TestCase
import unittest
from functools import partial

from easy_cfr.murder_mystery import MurderGameModel, MurderMysteryParams, MurderMysteryPlayer
from easy_cfr.policy_utils import get_policy_player, get_info_sets

# Todo: extend the set of unit tests to improve the test coverage
# also, some of the tests rely on knowing the number of information sets in a game - is there a better test for these cases?

class TestMurderGame(unittest.TestCase):

    def test_game_model_init_state(self):
        model = MurderGameModel()
        self.assertTrue(model.state.moves == [])
        self.assertTrue(not model.is_terminal())
        self.assertTrue(model.n_actions() == len(model.state.alive))
        self.assertTrue(model.is_chance_node())

    def test_game_model_state_one_allow_pass_and_suicide(self):
        params = MurderMysteryParams(allow_pass=True, allow_suicide=True)
        model = MurderGameModel()
        model.act(0)
        self.assertTrue(model.state.moves == [0])
        self.assertTrue(not model.is_terminal())
        self.assertTrue(model.n_actions() == 1 + len(model.state.alive))
        self.assertTrue(model.current_player() == MurderMysteryPlayer.KILLER)
        self.assertFalse(model.is_chance_node())

    def test_game_model_state_one_allow_pass(self):
        params = MurderMysteryParams(allow_pass=True, allow_suicide=False)
        model = MurderGameModel(params)
        model.act(0)
        self.assertTrue(model.state.moves == [0])
        self.assertTrue(not model.is_terminal())
        self.assertTrue(model.n_actions() == len(model.state.alive))
        self.assertTrue(model.current_player() == MurderMysteryPlayer.KILLER)
        self.assertFalse(model.is_chance_node())

    def test_game_model_state_one_deny_pass_and_suicide(self):
        params = MurderMysteryParams(allow_pass=False, allow_suicide=False)
        model = MurderGameModel(params)
        model.act(0)
        self.assertTrue(model.state.moves == [0])
        self.assertTrue(not model.is_terminal())
        self.assertTrue(model.n_actions() == len(model.state.alive) - 1)
        self.assertTrue(model.current_player() == MurderMysteryPlayer.KILLER)
        self.assertFalse(model.is_chance_node())

    def test_game_info_sets(self):
        params = MurderMysteryParams(allow_pass=False, allow_suicide=False)
        model_factory = partial(MurderGameModel, params)
        random_player = get_policy_player(model_factory, 0)
        policy = random_player.policy
        model = model_factory()
        info_sets = get_info_sets(model_factory(), {})
        for key, ix in info_sets.items():
            print(f"{key=}, {ix=}, {policy[ix]=}")
        self.assertEqual(len(info_sets), 103)



# with this uncommented none of the tests run - get "Empty suite"!
# unittest.main()

print("Running tests")


