

from abc import abstractmethod, ABC
from dataclasses import dataclass
import random
from typing import Set

from core.simple_game import Person, MurderGameState


@dataclass
class ObservationForKiller:
    alive: Set[Person]
    killer: Person

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForKiller(state.alive, state.killer)


@dataclass
class ObservationForDetective:
    alive: Set[Person]
    dead: Set[Person]
    accused: Set[Person]

    @classmethod
    def from_game_state(cls, state: MurderGameState):
        return ObservationForDetective(state.alive, state.dead, state.accused)

class KillerInterface(ABC):
    @abstractmethod
    def get_action(self, obs: ObservationForKiller) -> Person:
        pass


class DetectiveInterface(ABC):
    @abstractmethod
    def get_action(self, obs: ObservationForDetective) -> Person:
        pass

