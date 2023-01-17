
from __future__ import annotations

from typing import List, NamedTuple

import pyspiel

import numpy as np

import enum

N_PEOPLE = 4
ALLOW_PASS = True
PASS_ACTION = N_PEOPLE

ACTIONS = list(range(N_PEOPLE))
if ALLOW_PASS:
    ACTIONS.append(PASS_ACTION)

MAX_SCORE = N_PEOPLE * 10


class MurderMysteryParams(NamedTuple):
    allow_pass: bool = True
    allow_suicide: bool = True
    success_score: int = 100
    cost_per_accusation: int = 10
    cost_per_death: int = 10

class Rewards:
    SUCCESS: int = 100
    COST_PER_ACCUSATION: int = 10
    COST_PER_DEATH = 10


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
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(ACTIONS),
    max_chance_outcomes=N_PEOPLE,
    num_players=_NUM_PLAYERS,
    min_utility=-MAX_SCORE,
    max_utility=MAX_SCORE,
    utility_sum=0.0,
    max_game_length=2 * len(ACTIONS))  # e.g. Pass, Bet, Bet


class MurderMysteryPlayer(enum.IntEnum):
    KILLER = 0
    DETECTIVE = 1


class MurderMysteryVariationsGame(pyspiel.Game):
    """Very Simple Murder Mystery Game"""

    def __init__(self, params=None, game_params:MurderMysteryParams = None) -> None:
        self.game_params = game_params
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self, params: MurderMysteryParams = None) -> MurderMysteryVariationsState:
        """Returns a state corresponding to the start of a game."""
        return MurderMysteryVariationsState(self, params or self.game_params or MurderMysteryParams())

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return MurderMysteryVariationsObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class MurderMysteryVariationsState(pyspiel.State):
    """A python version of the Simon poker state."""

    def __init__(self, game, params: MurderMysteryParams = None):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.params = params
        self.alive = set(range(N_PEOPLE))
        self.dead = set()
        self.accused = set()
        self.killer = -1
        self._move_no = 0
        self._max_turns = 2 * N_PEOPLE

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
            action_set -= {self.killer}
        if self.params.allow_pass:
            action_set.add(PASS_ACTION)
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
        self.alive.discard(victim)
        self.dead.add(victim)

    def _accuse_action(self, suspect: int) -> None:
        self.accused.add(suspect)

    def _apply_action(self, action: int) -> None:
        """Applies the specified action to the state."""
        if action == PASS_ACTION:
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

    def is_terminal(self):
        """Returns True if the game is over i.e. max moves, or killer identified, or no one left alive except possibly the killer"""
        return (
                self._move_no >= self._max_turns or
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


    def __init__(self, iig, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation will depend on the player
        # The tensor comprises the following pieces given N players
        # Each set describes the killer identity (N+1), the alive people (N), the dead people(N), the accused (N)
        # The one-hot coding for killer has N elements, all will be zero if the killer is not assigned yet
        size = 4 * N_PEOPLE
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
            killer_list = self._code_killer(-1, N_PEOPLE)
        else:
            killer_list = self._code_killer(state.killer, N_PEOPLE)
        alive_list = self._code_set(state.alive, N_PEOPLE)
        dead_list = self._code_set(state.alive, N_PEOPLE)
        accused_list = self._code_set(state.alive, N_PEOPLE)
        all_list = [*killer_list, *alive_list, *dead_list, *accused_list]
        for i, x in enumerate(all_list):
            obs[i] = x
        print("All list: ",all_list)
        print("obs: ", obs)

    def string_from(self, state: MurderMysteryVariationsState, player: int):
        """Observation of `state` from the PoV of `player`, as a string."""
        if player == MurderMysteryPlayer.KILLER:
            return f"k_{state.killer}-a_{state.alive}"
        else:
            return f"a_{state.alive}_d{state.dead}_s{state.accused}"

# Register the game with the OpenSpiel library

# pyspiel.register_game(_GAME_TYPE, SimonPokerGame)

