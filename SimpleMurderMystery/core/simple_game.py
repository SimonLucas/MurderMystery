from __future__ import annotations

import copy
import dataclasses
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple, Set, Tuple, List


# We use a class here to group the reward constants to keep things neat
class Rewards:
    SUCCESS: int = 100
    COST_PER_ACCUSATION: int = 10
    COST_PER_DEATH = 10


# The idea of the Person NamedTuple is to store the immutable parts of a person: their id never changes\
# ah - but what about their location?  If the characters in a game get to move, we'd introduce a Vec2d class to
# manage easy movement on a grid
# A person can be a killer or a regular civilian
# But currently we model the detective as an external observer (e.g. the killer cannot kill the detective, and the detective has no location)

class Person(NamedTuple):
    id: int
    # location: Tuple[int, int]


# Using a dataclass instead of a NamedTuple to allow easy modification - may switch to a NamedTuple later
@dataclass
class MurderGameState:
    alive: Set[Person]
    dead: Set[Person]
    accused: Set[Person]
    killer: Person
    move_no: int = 0


def get_people(n_grid: int = 5, n_people: int = 8):
    locations = [(x, y) for x in range(n_grid) for y in range(n_grid)]
    random.shuffle(locations)
    # people = [Person(i, locations[i]) for i in range(n_people)]
    people = [Person(i) for i in range(n_people)]
    return people


class PartialStateRandomiser:
    def build_from_detective_observation(self, obs: ObservationForDetective) -> MurderGameState:
        possible_killers = (obs.alive - obs.dead) - obs.accused
        assert len(possible_killers) > 0
        guessed_killer = random.choice(list(possible_killers))
        state = MurderGameState(alive=obs.alive, dead=obs.dead, accused=obs.accused, killer=guessed_killer)
        return state


class MurderGameModel:
    def __init__(self, grid_size: int = 5, n_people: int = 5):
        people = get_people(grid_size, n_people)
        killer = random.choice(people)
        self.state = MurderGameState(alive=set(people), dead=set(), accused=set(), killer=killer)
        self.max_turns = 100

    # todo: decide whether to have sets of int or sets of persons
    # advantage of person is that we can specify their location

    def step_kill(self, victim: Person):
        self.state.alive.discard(victim)
        self.state.dead.add(victim)

    def step_accuse(self, suspect: Person):
        self.state.accused.add(suspect)
        # if killer

    def is_terminal(self):
        return self.state.move_no >= self.max_turns or self.n_actions() == 0

    def n_actions(self) -> int:
        if self.killer_turn():
            return len(self.state.alive - {self.state.killer})
        else:
            return len(self.state.alive - self.state.accused)

    def killer_turn(self):
        # alternate  between killer and detective moves, starting with killer
        return 0 == self.state.move_no % 2

    def act(self, action: Person) -> None:
        if self.killer_turn():
            self.step_kill(action)
        else:
            self.step_accuse(action)
        self.state.move_no += 1

    def copy_state(self) -> MurderGameModel:
        return copy.deepcopy(self)

    def score(self) -> float:
        total = 0
        if self.state.killer in self.state.accused:
            total += Rewards.SUCCESS
        total -= Rewards.COST_PER_DEATH * len(self.state.dead)
        # note the minus below is set minus - there  is no cost for accusing the killer
        total -= Rewards.COST_PER_ACCUSATION * len(self.state.accused - {self.state.killer})
        return total

    def get_actions(self):
        if self.killer_turn():
            return list(self.state.alive)
        else:
            return list(self.state.alive - self.state.accused)

