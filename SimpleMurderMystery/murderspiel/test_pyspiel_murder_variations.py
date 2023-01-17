from open_spiel.python.algorithms import exploitability, cfr

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame
import itertools as it



game = MurderMysteryVariationsGame()



def print_policy(policy):
    for state, probs in zip(it.chain(*policy.states_per_player),
                            policy.action_probability_array):
        print(f'{state:6}   p={probs}')


def get_cfr_policy(n: int):
    cfr_solver = cfr.CFRSolver(game)
    average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
        loss = exploitability.exploitability(game, average_policy)
        print(f"Exploitability ({i}) = {loss}")
    return average_policy


cfr_policy = get_cfr_policy(10)

print_policy(cfr_policy)
print(len(cfr_policy.action_probability_array))
