from __future__ import annotations

import copy
from typing import List, NamedTuple

import pyspiel

import numpy as np

import enum


# N_PEOPLE = 4
# ALLOW_PASS = True
# PASS_ACTION = N_PEOPLE

# ACTIONS = list(range(N_PEOPLE))
# if ALLOW_PASS:
#     ACTIONS.append(PASS_ACTION)

# MAX_SCORE = N_PEOPLE * 10


class MurderMysteryParams(NamedTuple):
    allow_pass: bool = True
    allow_suicide: bool = True
    success_score: int = 100
    cost_per_accusation: int = 10
    cost_per_death: int = 10
    n_people: int = 4
    max_turns: int = 7


class Rewards:
    SUCCESS: int = 100
    COST_PER_ACCUSATION: int = 10
    COST_PER_DEATH: int = 10


_NUM_PLAYERS = 2
_DECK = frozenset([0, 1, 2])
_GAME_TYPE = pyspiel.GameType(
    short_name="python_Murder_Mystery",
    long_name="Python Murder Mystery",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)


# _GAME_INFO = pyspiel.GameInfo(
#     num_distinct_actions=len(ACTIONS),
#     max_chance_outcomes=N_PEOPLE,
#     num_players=_NUM_PLAYERS,
#     min_utility=-MAX_SCORE,
#     max_utility=MAX_SCORE,
#     utility_sum=0.0,
#     max_game_length=2 * len(ACTIONS))  # e.g. Pass, Bet, Bet


class MurderMysteryPlayer(enum.IntEnum):
    KILLER = 0
    DETECTIVE = 1


class MurderMysteryVariationsGame(pyspiel.Game):
    """Very Simple Murder Mystery Game"""

    def __init__(self, params=None, game_params: MurderMysteryParams = None) -> None:
        game_params = game_params or MurderMysteryParams()
        self.game_params = game_params
        n_actions = game_params.n_people
        if game_params.allow_pass:
            n_actions += 1
        max_score = game_params.n_people * 10
        game_info = pyspiel.GameInfo(
            num_distinct_actions=n_actions,
            max_chance_outcomes=game_params.n_people,
            num_players=2,
            min_utility=-max_score,
            max_utility=max_score,
            utility_sum=0.0,
            max_game_length=game_params.max_turns
        )
        super().__init__(_GAME_TYPE, game_info, params or dict())

    def new_initial_state(self, params: MurderMysteryParams = None) -> MurderMysteryVariationsState:
        """Returns a state corresponding to the start of a game."""
        return MurderMysteryVariationsState(self, self.game_params)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return MurderMysteryVariationsObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            self.game_params)


class MurderMysteryVariationsState(pyspiel.State):
    """A python version of the Simon poker state."""

    def __init__(self, game, params: MurderMysteryParams = None):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.params = params
        self.alive = set(range(params.n_people))
        self.dead = set()
        self.accused = set()
        self.killer = -1
        self._move_no = 0
        self.pass_action = params.n_people

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        # in the current version only the first move is a chance move - the one which decides who is the killer
        elif self._move_no == 0:
            return pyspiel.PlayerId.CHANCE
        else:
            return self.to_play()

    def to_play(self) -> int:
        return 1 - (self._move_no % 2)

    def _legal_actions(self, player) -> List[int]:
        """Returns a list of legal actions, sorted in ascending order (actually not sure about the order)"""
        # check this is not the chance player, for some reason that is handled separately
        assert player >= 0
        action_set = set(self.alive)
        if self.current_player() == MurderMysteryPlayer.KILLER and not self.params.allow_suicide:
            print(f"Removing killer {self.killer} from actions {action_set}")
            action_set -= {self.killer}
        if self.params.allow_pass:
            action_set.add(self.pass_action)
        else:
            assert self.pass_action not in action_set
        print(f"Returning action set: {action_set}")
        return list(action_set)

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities.
            The only chance outcome is at the start of the game when the killer id is randomly chosen
        """
        assert self._move_no == 0
        # at this stage all players are alive, just allocate the killer as one of them
        outcomes = list(self.alive)
        p = 1.0 / len(outcomes)
        return [(o, p) for o in outcomes]

    def _kill_action(self, victim: int) -> None:
        print(f" {self.killer} to kill {victim}, {self.alive}")
        assert self.params.allow_suicide == False
        assert self.params.allow_pass == False
        assert self.current_player() == MurderMysteryPlayer.KILLER
        if not self.params.allow_suicide:
            assert not victim == self.killer, f"{victim}, {self.killer}, {self.alive}, {self._move_no}, {self.current_player()}, {self._legal_actions(self.current_player())}"
        self.alive.discard(victim)
        self.dead.add(victim)

    def _accuse_action(self, suspect: int) -> None:
        assert self.current_player() == MurderMysteryPlayer.DETECTIVE
        self.accused.add(suspect)

    def _apply_action(self, action: int) -> None:
        """Applies the specified action to the state."""
        if action == self.pass_action:
            assert self.params.allow_pass
            self._move_no += 1
            return

        if self.is_chance_node():
            # the action is the id of the selected individual, we just allocate this as the killer, no other changes
            assert self._move_no == 0
            self.killer = action
        else:
            if self.current_player() == MurderMysteryPlayer.DETECTIVE:
                self._accuse_action(action)
            else:
                self._kill_action(action)
        self._move_no += 1

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Killer is: {action}"
        elif player == MurderMysteryPlayer.DETECTIVE:
            return f"Accuse: {action}"
        else:
            return f"Kill: {action}"

    def n_actions(self) -> int:
        if self._move_no == 0:
            return len(self.alive)
        else:
            return len(self._legal_actions(self.current_player()))

    # def check_sanity(self) -> None:
    #     assert

    def clone(self):
        cp = super().clone()
        return cp

    def is_terminal(self):
        """Returns True if the game is over i.e. max moves, or killer identified, or no one left alive except possibly the killer"""
        return (
                self._move_no >= self.params.max_turns or
                len(self.alive - {self.killer}) == 0 or
                self.killer in self.accused
        )

    def score(self) -> float:
        if not self.is_terminal():
            return 0

        total = 0
        if self.killer in self.accused:
            total += self.params.success_score
        total -= self.params.cost_per_death * len(self.dead)
        # note the minus below is set minus - there  is no cost for accusing the killer
        total -= self.params.cost_per_accusation * len(self.accused - {self.killer})
        # if total += Rewards.SUICIDE_PENALTY
        return total

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self.is_terminal():
            return [0, 0]
        else:
            score = self.score()
            return [score, -score]

    def __str__(self):
        # todo: update this
        """String for debug purposes. No particular semantics are required."""
        return f"k={self.killer}, a={self.alive}, s={self.accused}, m={self._move_no}"


class MurderMysteryVariationsObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig, params: MurderMysteryParams):
        """Initializes an empty observation tensor."""
        if params == None:
            raise ValueError(f"Observation needs params for setup; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation will depend on the player
        # The tensor comprises the following pieces given N players
        # Each set describes the killer identity (N), the alive people (N), the dead people(N), the accused (N)
        # The one-hot coding for killer has N elements, all will be zero if the killer is not assigned yet
        self.params = params
        size = 4 * params.n_people
        shape = (size)
        self.tensor = np.zeros(size, np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def _code_set(self, people: set, n: int) -> List[int]:
        return [1 if x in people else 0 for x in range(n)]

    def _code_killer(self, killer: int, n: int) -> List[int]:
        return [1 if x == killer else 0 for x in range(n)]

    def set_from(self, state: MurderMysteryVariationsState, player: int):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        if player == MurderMysteryPlayer.DETECTIVE:
            killer_list = self._code_killer(-1, self.params.n_people)
        else:
            killer_list = self._code_killer(state.killer, self.params.n_people)
        alive_list = self._code_set(state.alive, self.params.n_people)
        dead_list = self._code_set(state.alive, self.params.n_people)
        accused_list = self._code_set(state.alive, self.params.n_people)
        all_list = [*killer_list, *alive_list, *dead_list, *accused_list]
        for i, x in enumerate(all_list):
            obs[i] = x
        # print("All list: ", all_list)
        # print("obs: ", obs)

    def string_from(self, state: MurderMysteryVariationsState, player: int):
        """Observation of `state` from the PoV of `player`, as a string."""
        if player == MurderMysteryPlayer.KILLER:
            return f"k_{state.killer}-a_{state.alive}"
        else:
            return f"a_{state.alive}_d{state.dead}_s{state.accused}"

# Register the game with the OpenSpiel library

# pyspiel.register_game(_GAME_TYPE, SimonPokerGame)
