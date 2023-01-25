from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto
from typing import Optional, Union, List, Tuple

import pygame
from pygame.locals import *
# from pygame.locals import (
#     K_UP,
#     K_DOWN,
#     K_LEFT,
#     K_RIGHT,
#     K_SPACE,
#     QUIT,
# )


class Shape(Enum):
    CIRCLE = auto()
    DISC = auto()
    RECT = auto()
    DIAMOND = auto()


class GraphView:
    SIZE = 25
    BORDER = 1
    COLORS = (
        (128, 128, 255),
        (255, 128, 128),
        (128, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (90, 128, 30),
        (200, 128, 64),
        (250, 200, 180),
    )

    def __init__(self, graph: GraphNode, params: Params) -> None:
        pygame.init()
        # Set up the drawing window
        self.graph = graph
        self.depth_lut = graph.get_depth_dict()
        self.params = params
        self.screen = pygame.display.set_mode([params.width, params.height], RESIZABLE)
        self.clock = pygame.time.Clock()

    def run(self):
        running: bool = True

        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            pressed_keys = pygame.key.get_pressed()
            self.process_keys(pressed_keys)
            self.draw_background()
            self.draw_graph()

            # Flip the display
            pygame.display.flip()
            self.clock.tick(50)

        # Done! Time to quit.
        pygame.quit()

    def process_keys(self, pressed_keys):
        pass

    def draw_background(self):
        # Fill the background with white
        self.screen.fill(self.COLORS[1])

    def diamond(self, size, cx, cy) -> List[Tuple[float, float]]:
        return [
            (cx, cy - size),
            (cx + size, cy),
            (cx, cy + size),
            (cx - size, cy),
        ]

    def draw_graph(self):
        max_nodes_in_layer = max(len(v) for k, v in self.depth_lut.items())
        width = self.screen.get_width()
        height = self.screen.get_height()
        x_gap_max = width / max_nodes_in_layer
        rad = x_gap_max * 0.8
        cx = width / 2
        y_gap = height / len(self.depth_lut.keys())

        for k, v in self.depth_lut.items():
            y = (k + 0.5) * y_gap
            x_gap = width / len(v)
            for i, node in enumerate(v):
                x = (i + 0.5) * x_gap
                node.pos = (x, y)

        edges = self.graph.get_edges()
        for edge in edges:
            p0 = edge[0].pos
            p1 = edge[1].pos
            pygame.draw.line(self.screen, self.COLORS[3], p0, p1, width=5)
            mid_point = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1])/2)
            font = pygame.font.SysFont(None, 25)
            edge_label = str(edge[2])
            # print(edge_label)
            text = font.render(edge_label, True, (50, 50, 50), (200, 200, 200))
            text_rect = text.get_rect(center=mid_point)
            self.screen.blit(text, text_rect)

        for node in self.graph.get_nodes():
            shape = node.get_shape()
            if shape == Shape.DIAMOND:
                poly = self.diamond(rad/2, *node.pos)
                pygame.draw.polygon(self.screen, self.COLORS[4], poly, 0)
            elif shape == Shape.DISC:
                pygame.draw.circle(self.screen, self.COLORS[5], node.pos, rad/2, 0)
            elif shape == Shape.CIRCLE:
                pygame.draw.circle(self.screen, self.COLORS[7], node.pos, rad/2, 0)
            else:
                x,y = node.pos
                rect = (x - rad/2, y - rad/2, rad, rad)
                pygame.draw.rect(self.screen, self.COLORS[2], rect)
            font = pygame.font.SysFont(None, 30)
            text = font.render(str(node.label or node.id), True, (200, 0, 0))
            text_rect = text.get_rect(center=node.pos)
            self.screen.blit(text, text_rect)


class GraphNode:
    def __init__(self, id_gen: Inc, depth: int):
        self.id_gen = id_gen
        self.id = id_gen()
        self.depth = depth
        self.children = []
        self.edge_labels = []
        self.pos = (0, 0)
        self.label = None

    def add(self, edge_label: str = "") -> GraphNode:
        child = GraphNode(self.id_gen, self.depth + 1)
        self.children.append(child)
        self.edge_labels.append(edge_label)
        return child

    def print(self):
        print(f"{self.id=}, {self.depth=}")
        for x in self.children: x.print()

    def get_depth_dict(self, d: Optional[dict] = None) -> dict:
        d = d or defaultdict(list)
        d[self.depth].append(self)
        for x in self.children: x.get_depth_dict(d)
        return d

    def get_edges(self, l: Optional[list] = None) -> list:
        l = l or []
        for node, edge_label in zip(self.children, self.edge_labels):
            l.append((self, node, edge_label))
            node.get_edges(l)
        return l

    def get_nodes(self, l: Optional[list] = None) -> list:
        l = l or []
        l.append(self)
        for x in self.children:
            x.get_nodes(l)
        return l

    def is_terminal(self) -> bool:
        return len(self.children) == 0

    def get_shape(self) -> Shape:
        if self.is_terminal():
            return Shape.DIAMOND
        elif self.depth == 0:
            return Shape.DISC
        elif self.depth % 2 == 0:
            return Shape.CIRCLE
        else:
            return Shape.RECT


class Params:
    def __init__(self):
        self.width = 1200
        self.height = 700


def build_graph(node: Union[object, dict], parent: GraphNode) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            child: GraphNode = parent.add()
            build_graph(v, child)


# def get_graph() -> GraphNode:
#     root = GraphNode(id_gen=Inc(), depth=0)
#     build_graph(kuhn_tree, root)
#     root.print()
#     print(root.get_depth_dict())
#     return root
#
#
# def main():
#     graph = get_graph()
#     GraphView(graph, Params()).run()
#     # print(get_graph())
#
#
# if __name__ == '__main__':
#     main()
