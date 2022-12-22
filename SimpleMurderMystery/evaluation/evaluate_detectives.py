import statistics
from typing import List
from scipy import stats

from agents.random_agents import RandomKillerWithSuicide, RandomKillerNoSuicide, RandomDetective
from core.game_interfaces import DetectiveInterface, ObservationForKiller, ObservationForDetective
from core.simple_game import MurderGameModel


class MurderGameController:
    def __init__(self, model: MurderGameModel, detective: DetectiveInterface):
        self.model = model
        self.players = [RandomKillerWithSuicide(), detective]
        self.players = [RandomKillerNoSuicide(), detective]
        # these are the constructors for each observation instance, will be called with each MurderGameState object
        self.observers = [
            ObservationForKiller.from_game_state,
            ObservationForDetective.from_game_state,
        ]

    def run_game(self) -> float:
        while not self.model.is_terminal():
            # print()
            player_index = self.model.state.move_no % len(self.players)
            observer = self.observers[player_index]
            # print(observer)
            observation = observer(self.model.state)

            to_play = self.players[player_index]
            action = to_play.get_action(observation)
            self.model.act(action)
        return self.model.score()


def run_evaluations(n_trials: int = 10, detective: DetectiveInterface = RandomDetective()) -> List[float]:
    results = []
    for _ in range(n_trials):
        controller = MurderGameController(MurderGameModel(grid_size=10, n_people=7), detective)
        result = controller.run_game()
        results.append(result)
    return results


if __name__ == '__main__':
    results = run_evaluations(1000, RandomDetective())
    print(f"{statistics.mean(results)=}")
    print(f"{statistics.stdev(results)=}")
    print(f"{stats.sem(results)=}")
    print(f"{len(results)=}")

# todo: update the action space to allow for no_kills

# todo: implement the RollingHorizon Detective Agent