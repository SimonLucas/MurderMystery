import logging
from typing import List
from pyspiel import TabularPolicy

from murderspiel.pyspiel_utilities import get_cfr_policy
from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams
from open_spiel.python import policy as policy_lib

from murderspiel.pyspiel_utilities import policy_as_list, sample

# set up a logger so that we can easily turn on/off turn by turn printing
game_logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)

game_logger.addHandler(c_handler)


# this should work for any openspiel game, but I've been lazy and put this type for now
def run_game(game: MurderMysteryVariationsGame, players: List[TabularPolicy]) -> List[float]:
    state = game.new_initial_state()
    print(state.params)
    while not state.is_terminal():
        game_logger.info(f"\nMove number: {state.move_number()}")
        game_logger.info(f"State: {state}")
        if state.is_chance_node():
            action_probs = state.chance_outcomes()
        else:
            player_policy = players[state.current_player()]
            action_probs = policy_as_list(player_policy, state)
        action = sample(action_probs)
        game_logger.info(f"Player {state.current_player()} took action {action}")
        state = state.child(action)
        game_logger.info(f"New state -> : {state}")
    game_logger.info(f"\nFinal state: {state}")
    return state.returns()


def run_games(n_games: int = 1, cfr_iterations=10) -> None:
    params = MurderMysteryParams(allow_pass=False, allow_suicide=False)
    game = MurderMysteryVariationsGame(game_params=params)
    # create a uniform random policy
    game_logger.info(game.game_params)
    uniform_random_policy = policy_lib.TabularPolicy(game)
    # cfr_policy = get_cfr_policy(game, cfr_iterations)
    player_policies = [uniform_random_policy, uniform_random_policy]
    total = 0
    for i in range(n_games):
        # zero sum game, just take return for first player
        result = run_game(game, player_policies)[0]
        game_logger.debug(f"{i}: \t {result}")
        total += result
    print(f"Average: {total / n_games:1}")


if __name__ == '__main__':
    # (un)comment following to turn (on/off) logging
    # game_logger.setLevel(logging.INFO)
    game_logger.info("Test message")
    run_games(1000)
