from functools import partial

from easy_cfr.game_state import GameState, Player
from easy_cfr.murder_mystery import MurderGameModel, MurderMysteryParams

from easy_cfr.policy_utils import PolicyPlayer, get_policy_player


def evaluate(state: GameState, player: PolicyPlayer, opponent: PolicyPlayer, player_role: int):
    if state.is_terminal():
        return state.returns()[player_role]
    elif state.current_player() == player_role:
        ap = player.get_action_probs(state)
    elif state.current_player() == Player.CHANCE:
        ap = state.chance_action_probs()
    else:
        ap = opponent.get_action_probs(state)
    return sum(p * evaluate(state.child(a), player, opponent, player_role) for a, p in ap)


def eval(state_factory, player: PolicyPlayer, opponent: PolicyPlayer):
    results = (evaluate(state_factory(), player, opponent, player_role=0),
               + evaluate(state_factory(), player, opponent, player_role=1))
    print(f"{results=}")
    return sum(results)


def print_eval(state_factory, player: PolicyPlayer, opponent: PolicyPlayer) -> None:
    print("Policy =:")
    print(player.policy)
    print("Opponent =:")
    print(opponent.policy)
    print()
    score = eval(state_factory, player, opponent)
    print(f"{score=}")
    print()


if __name__ == '__main__':
    # state_factory = KuhnPoker

    params = MurderMysteryParams(allow_pass=False, allow_suicide=True, n_people=4, max_turns=8)
    state_factory = partial(MurderGameModel, params)


    policy_player = get_policy_player(state_factory, n_iterations=100)

    # calling with zero iterations results in a uniform random policy
    random_player = get_policy_player(state_factory, n_iterations=0)

    # print(f"{policy_player=}")
    print()
    print("Random player", random_player.policy)
    print(f"{state_factory=}")
    print_eval(state_factory, policy_player, random_player)
    print_eval(state_factory, policy_player, random_player)
