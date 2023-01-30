import copy
from enum import IntEnum

# class KuhnPoker:


# todo: deck
# todo: current_player
# todo: is chance_node
# todo: state
from typing import List

from easy_cfr.game_and_agent_interfaces import GameModel, Player

_DECK = frozenset([0, 1, 2])
_NUM_PLAYERS = 2

N_STATE_TRANSITIONS = 0


class Action(IntEnum):
    PASS = 0
    BET = 1


class KuhnPoker(GameModel):
    """
    This is included here for testing purposes - easy to cross-check the correct CFR values
    """

    def __init__(self):
        self.cards = []
        self.bets = []
        self.pot = [1.0, 1.0]
        self.game_over: bool = False

    def is_terminal(self) -> bool:
        return self.game_over

    def current_player(self) -> int:
        if self.is_terminal():
            return Player.TERMINAL
        elif len(self.cards) < _NUM_PLAYERS:
            return Player.CHANCE
        else:
            return len(self.bets) % _NUM_PLAYERS

    def n_actions(self) -> int:
        return len(self.actions())

    def max_actions(self) -> int:
        return 2

    def actions(self) -> List[int]:
        if self.current_player() == Player.CHANCE:
            return self.chance_actions()
        else:
            return self.player_actions()

    def act(self, action: int) -> None:
        global N_STATE_TRANSITIONS
        N_STATE_TRANSITIONS += 1
        if self.current_player() == Player.CHANCE:
            self.cards.append(action)
        else:
            assert action in {Action.PASS, Action.BET}
            if action == Action.BET:
                self.pot[self.current_player()] += 1
            # this will also advance self.current_player()
            self.bets.append(action)
            if (min(self.pot) == 2 or
                    (len(self.bets) == 2 and action == Action.PASS) or
                    len(self.bets) == 3):
                self.game_over = True

    def total_state_transitions(self) -> int:
        return N_STATE_TRANSITIONS

    def action_to_string(self, action):
        """Action -> string."""
        if self.current_player() == Player.CHANCE:
            return f"Dl:{action}"
        elif action == Action.PASS:
            return "Pass"
        else:
            return "Bet"

    def returns(self) -> List[float]:
        pot = self.pot
        winnings = float(min(pot))
        if not self.is_terminal():
            return [0., 0.]
        elif pot[0] > pot[1]:
            return [winnings, -winnings]
        elif pot[0] < pot[1]:
            return [-winnings, winnings]
        elif self.cards[0] > self.cards[1]:
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]

    def chance_actions(self) -> List[int]:
        assert self.current_player() == Player.CHANCE
        return sorted(_DECK - set(self.cards))

    def player_actions(self) -> List[int]:
        return [Action.PASS, Action.BET]

    def copy_state(self) -> GameModel:
        return copy.deepcopy(self)

    def information_set(self) -> str:
        if self.is_terminal() or (self.current_player() == Player.CHANCE):
            # return f"{self.current_player()}, {self.cards}"
            return ""
        else:
            player = self.current_player()
            visible_card = str(self.cards[player])
            return visible_card + "".join(["pb"[b] for b in self.bets])

    def __str__(self):
        if self.is_terminal():
            return f"{self.returns()[0]}"
        else:
            return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])

    def history(self) -> str:
        return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])


if __name__ == '__main__':
    state = KuhnPoker()
    print(f"{state.current_player()=}")
    print(f"{state.chance_actions()=}")
    state.act(state.actions()[0])
    print(f"{state.chance_actions()=}")
    state.act(state.actions()[0])
    # print(f"{state.chance_actions()=}")
