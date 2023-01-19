from open_spiel.python.algorithms import exploitability, cfr

from murderspiel.pyspiel_murder_variations import MurderMysteryVariationsGame, MurderMysteryParams


def get_cfr_policy(game: MurderMysteryVariationsGame, n: int):
    cfr_solver = cfr.CFRSolver(game)
    average_policy = None
    for i in range(n):
        cfr_solver.evaluate_and_update_policy()
        average_policy = cfr_solver.average_policy()
        # loss = exploitability.exploitability(game, average_policy)
        # print(f"Exploitability ({i}) = {loss}")
    return average_policy
