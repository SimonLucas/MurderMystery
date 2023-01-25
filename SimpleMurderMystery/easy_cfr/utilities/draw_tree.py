from typing import List, Optional

from easy_cfr.game_state import GameState
from easy_cfr.utilities.graph_view_pygame import GraphNode, GraphView, Params


def enumerate_states(state: GameState, states: Optional[List[GameState]] = None) -> List[GameState]:
    states = states or []
    states.append(state)
    if not state.is_terminal():
        for action in state.actions():
            next_state = state.copy_state()
            next_state.act(action)
            enumerate_states(next_state, states)
    return states


def build_graph(state: GameState, parent: Optional[GraphNode] = None) -> GraphNode:
    parent.label = state.information_set()  # str(state)
    parent.label = str(state)
    print(f"{parent.label=}, {state.information_set()=}")
    if not state.is_terminal():
        for action in state.actions():
            action_string = state.action_to_string(action)
            next_state = state.copy_state()
            next_state.act(action)
            child = parent.add(action_string)
            child.label = str(state)
            build_graph(next_state, child)
    return parent


class Inc:
    # just provide an easy way to increment a counter to label a state
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, *args, **kwargs) -> int:
        self.n += 1
        return self.n

    def label(self, s: str) -> str:
        self.n += 1
        return f"{s}-{self.n}"


def get_graph(state_factory) -> GraphNode:
    root = GraphNode(id_gen=Inc(), depth=0)
    build_graph(state_factory(), root)
    root.print()
    print(root.get_depth_dict())
    return root


def draw_game_tree(state_factory):
    graph = get_graph(state_factory)
    GraphView(graph, Params()).run()
