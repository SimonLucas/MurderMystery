from core.game_interfaces import KillerInterface, ObservationForKiller, DetectiveInterface, ObservationForDetective
from core.simple_game import Person
import random


class RandomKillerWithSuicide(KillerInterface):
    # just pick a victim at random, including oneself!
    def get_action(self, obs: ObservationForKiller) -> Person:
        return random.choice(list(obs.alive))


class RandomKillerNoSuicide(KillerInterface):
    # just pick a victim at random, including oneself!
    def get_action(self, obs: ObservationForKiller) -> Person:
        return random.choice(list(obs.alive - {obs.killer}))


# todo: implement a random killer that will not suicide

# the random detective picks a non-accused alive person each time
class RandomDetective(DetectiveInterface):
    def get_action(self, obs: ObservationForDetective) -> Person:
        # print(f"{obs=}")
        # print(f"{(obs.alive - obs.accused)=}")
        return random.choice(list(obs.alive - obs.accused))


# a dumb detective that accuses the same more than once, and even dead people
class RanDumbDetective(DetectiveInterface):
    def get_action(self, obs: ObservationForDetective) -> Person:
        return random.choice(list(obs.alive | obs.dead))
