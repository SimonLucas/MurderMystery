from dataclasses import dataclass
from typing import List, NamedTuple, Set
from enum import IntEnum
from functools import partial

import random
import copy

from easy_cfr.game_state import GameState, Player

N_STATE_TRANSITIONS: int = 0

import logging

game_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
# change level here to turn logging on or off for this file
c_handler.setLevel(logging.INFO)
game_logger.addHandler(c_handler)


class MurderMysteryPlayer(IntEnum):
    KILLER = 0
    DETECTIVE = 1


class MurderMysteryParams(NamedTuple):
    allow_pass: bool = True
    allow_suicide: bool = True
    n_people: int = 4
    max_turns: int = 7
    cost_per_accusation: int = 10
    cost_per_death: int = 10
    success_score: int = 100


# Using a dataclass instead of a NamedTuple to allow easy modification - may switch to a NamedTuple later
@dataclass
class MurderGameState:
    alive: Set[int]
    dead: Set[int]
    accused: Set[int]
    killer: int
    moves: List[int]


class MurderGameModel(GameState):
    def __init__(self, params: MurderMysteryParams = None):
        self.params = params or MurderMysteryParams()
        people = list(range(self.params.n_people))
        killer = -1
        self.state = MurderGameState(alive=set(people), dead=set(), accused=set(), killer=killer, moves=[])
        self.pass_action = -1

    def move_no(self) -> int:
        return len(self.state.moves)

    def is_terminal(self) -> bool:
        return (
                self.move_no() >= self.params.max_turns or
                self.state.killer in self.state.accused or
                len(self.state.alive - {self.state.killer}) == 0
        )

    def current_player(self) -> int:
        if self.is_terminal():
            return Player.TERMINAL
        elif self.move_no() == 0:
            return Player.CHANCE
        else:
            return 1 - (self.move_no() % 2)

    def is_chance_node(self) -> bool:
        return self.current_player() == Player.CHANCE

    def chance_actions(self) -> List[int]:
        assert self.current_player() == Player.CHANCE
        return list(range(len(self.state.alive)))

    def player_actions(self, player: int) -> List[int]:
        """Returns a list of legal actions, sorted in ascending order (actually not sure about the order)"""
        # check this is not the chance player, for some reason that is handled separately
        assert player >= 0
        action_set = set(self.state.alive)
        if self.current_player() == MurderMysteryPlayer.KILLER and not self.params.allow_suicide:
            game_logger.info(f"Removing killer {self.state.killer} from actions {action_set}")
            action_set -= {self.state.killer}
        if self.params.allow_pass:
            action_set.add(self.pass_action)
        else:
            assert self.pass_action not in action_set
        game_logger.info(f"Returning action set: {action_set}")
        return list(action_set)

    def actions(self) -> List[int]:
        if self.current_player() == Player.CHANCE:
            return self.chance_actions()
        else:
            return self.player_actions(self.current_player())

    def n_actions(self) -> int:
        return len(self.actions())

    def max_actions(self) -> int:
        n_max = self.params.n_people
        if self.params.allow_pass:
            n_max += 1
        return n_max

    def kill_action(self, victim: int) -> None:
        if self.state.killer not in self.state.alive:
            # the killer can't make a kill if they are already dead
            return
        game_logger.info(f" {self.state.killer} to kill {victim}, {self.state.alive}")
        assert self.current_player() == MurderMysteryPlayer.KILLER
        if not self.params.allow_suicide:
            assert not victim == self.state.killer, f"{victim}, {self.state.killer}, {self.state.alive}, {self.move_no()}, {self.current_player()}"
        self.state.alive.discard(victim)
        self.state.dead.add(victim)

    def accuse_action(self, suspect: int) -> None:
        assert self.current_player() == MurderMysteryPlayer.DETECTIVE
        self.state.accused.add(suspect)

    def act(self, action: int) -> None:
        global N_STATE_TRANSITIONS
        N_STATE_TRANSITIONS += 1
        if action == self.pass_action:
            assert self.params.allow_pass
            self.state.moves.append(action)
            return

        if self.is_chance_node():
            # the action is the id of the selected individual, we just allocate this as the killer, no other changes
            assert self.move_no() == 0
            self.state.killer = action
        else:
            if self.current_player() == MurderMysteryPlayer.DETECTIVE:
                self.accuse_action(action)
            else:
                assert self.current_player() == MurderMysteryPlayer.KILLER
                self.kill_action(action)
        self.state.moves.append(action)

    def total_state_transitions(self) -> int:
        return N_STATE_TRANSITIONS

    def score(self) -> float:
        if not self.is_terminal():
            return 0

        total = 0
        if self.state.killer in self.state.accused:
            total += self.params.success_score
        total -= self.params.cost_per_death * len(self.state.dead)
        # note the minus below is set minus - there  is no cost for accusing the killer
        total -= self.params.cost_per_accusation * len(self.state.accused - {self.state.killer})
        # if total += Rewards.SUICIDE_PENALTY
        return total

    def returns(self) -> List[float]:
        if not self.is_terminal():
            return [0, 0]
        score = self.score()
        return [score, -score]

    def copy_state(self) -> GameState:
        return copy.deepcopy(self)

    def information_set(self) -> str:
        if self.is_terminal():
            return "."
        elif self.current_player() == Player.CHANCE:
            return "?"
        else:
            # player = self.current_player()
            # miss the first one, as that's a chance node that must be excluded from the information set
            s = "?"
            if len(self.state.moves) > 0:
                s += str(self.state.moves[1:])
            return s

    def current_player_string(self):
        lut = {-1: "C", 0: "K", 1: "D"}
        return lut[self.current_player()]

    def action_to_string(self, action):
        """Action -> string."""
        if self.current_player() == Player.CHANCE:
            return f"C: {action}"
        elif action == self.pass_action:
            return f"{self.current_player_string()}:P"
        else:
            return f"{self.current_player_string()}:{action}"

    def __str__(self):
        if self.is_terminal():
            return f"{self.returns()[0]}"
        else:
            k = self.state.killer
            return f"{k=}," + str(self.state.moves)

    def history(self) -> str:
        return "/" + "".join(["-"[m] for m in self.state.moves])


if __name__ == '__main__':
    params = MurderMysteryParams(allow_pass=True)
    game = MurderGameModel()
    print(game.is_terminal())
    print(game.information_set())
    game.act(0)
    print(game.information_set())
    game.act(0)
    print(game.information_set())
    game.act(1)
    print(game.information_set())
