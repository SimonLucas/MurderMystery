# from unittest import main, TestCase
import logging
import unittest
from functools import partial

from easy_cfr.game_state import Player
from easy_cfr.murder_mystery import MurderGameModel, MurderMysteryParams, MurderMysteryPlayer
from easy_cfr.policy_utils import get_policy_player, get_info_sets
from easy_cfr.simpler_cfr import InfoSetTabularPolicy, run_easy_cfr, TabularPolicyPlayer


# Todo: extend the set of unit tests to improve the test coverage
# also, some of the tests rely on knowing the number of information sets in a game - is there a better test for these cases?

class TestMurderGamePolicyActions(unittest.TestCase):

    def test_game_action_space(self):
        n_people = 4
        params = MurderMysteryParams(allow_pass=True, allow_suicide=False, n_people=n_people, max_turns=8)
        model = MurderGameModel(params)
        print(model.n_actions())
        print(model.actions())
        self.assertTrue(model.is_chance_node())
        self.assertEqual(n_people, model.n_actions())
        self.assertEqual(model.current_player(), Player.CHANCE)
        model.act(0)
        print(model.n_actions())
        print(model.actions())
        self.assertEqual(n_people, model.n_actions())
        self.assertFalse(model.is_chance_node())
        self.assertNotIn(model.state.killer, set(model.actions()))
        self.assertEqual(model.current_player(), MurderMysteryPlayer.KILLER)
        model.act(1)
        print(model.n_actions())
        print(f"{model.actions()=}")
        # self.assertEqual(n_people-1, model.n_actions())
        print(f"{model.state.killer=}")
        self.assertEqual(model.current_player(), MurderMysteryPlayer.DETECTIVE)
        self.assertIn(model.state.killer, set(model.actions()))

    def test_game_action_distributions(self):
        n_people = 4
        params = MurderMysteryParams(allow_pass=True, allow_suicide=True, n_people=n_people, max_turns=8)
        model = MurderGameModel(params)
        policy = run_easy_cfr(partial(MurderGameModel, params), 3)
        player = TabularPolicyPlayer(policy)
        print(model.n_actions())
        print(model.actions())
        self.assertTrue(model.is_chance_node())
        self.assertEqual(model.current_player(), Player.CHANCE)
        model.act(0)
        print(model.n_actions())
        print(model.actions())
        self.assertFalse(model.is_chance_node())
        self.assertEqual(model.current_player(), MurderMysteryPlayer.KILLER)
        # print(f"{policy.policy_dict=}")
        print(f"{player.get_action_probs(model)=}")
        model.act(1)
        print(f"{model.actions()=}")
        # self.assertEqual(n_people-1, model.n_actions())
        print(f"{model.state.killer=}")
        print(f"{player.get_action_probs(model)=}")
        self.assertEqual(model.current_player(), MurderMysteryPlayer.DETECTIVE)


print("Running tests")
