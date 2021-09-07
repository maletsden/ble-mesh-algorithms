import numpy as np
import itertools
from typing import List, FrozenSet


def k2_pruning(adjacency_matrix: np.ndarray,
               min_reduction_neighbour: int = 3,
               max_reduction_neighbour: int = 4) -> np.ndarray:
    roles = np.zeros((len(adjacency_matrix),), dtype='i8')
    nodes_neighbours: List[FrozenSet[int]] = [neighbours_frozenset(row) for row in adjacency_matrix]
    k2_pruning_construct(adjacency_matrix, roles, nodes_neighbours)
    k2_pruning_first_reduction(adjacency_matrix, roles, nodes_neighbours)
    k2_pruning_second_reduction(adjacency_matrix, roles, nodes_neighbours)
    k2_pruning_second_reduction(adjacency_matrix, roles, nodes_neighbours,
                                min_reduction_neighbour=min_reduction_neighbour,
                                max_reduction_neighbour=max_reduction_neighbour)
    return roles


def k2_pruning_construct(adjacency_matrix: np.ndarray, roles: np.ndarray,
                         nodes_neighbours: List[FrozenSet[int]] = None) -> None:
    if nodes_neighbours is None:
        nodes_neighbours: List[FrozenSet[int]] = [neighbours_frozenset(row) for row in adjacency_matrix]

    for node, node_neighbours in enumerate(nodes_neighbours):
        for neighbour in node_neighbours:
            neighbour_neighbours: FrozenSet[int] = nodes_neighbours[neighbour].union(frozenset([neighbour]))

            if not node_neighbours.issubset(neighbour_neighbours):
                roles[node] = 1
                break


def k2_pruning_first_reduction(adjacency_matrix: np.ndarray, roles: np.ndarray,
                               nodes_neighbours: List[FrozenSet[int]] = None) -> None:
    if nodes_neighbours is None:
        nodes_neighbours: List[FrozenSet[int]] = [neighbours_frozenset(row) for row in adjacency_matrix]

    marked_nodes: List[int] = [node for node in range(len(roles)) if roles[node]]

    for node in marked_nodes:
        node_neighbours: FrozenSet[int] = nodes_neighbours[node]
        node_marked_neighbours: List[int] = [node for node in node_neighbours]

        for neighbour_node in node_marked_neighbours:
            neighbour_neighbours: FrozenSet[int] = nodes_neighbours[neighbour_node].union(frozenset([neighbour_node]))

            if not node_neighbours.issubset(neighbour_neighbours):
                continue

            roles[node] = 0
            break


def k2_pruning_second_reduction(adjacency_matrix: np.ndarray, roles: np.ndarray,
                                nodes_neighbours: List[FrozenSet[int]] = None,
                                min_reduction_neighbour: int = 2,
                                max_reduction_neighbour: int = 2) -> None:
    if nodes_neighbours is None:
        nodes_neighbours: List[FrozenSet[int]] = [neighbours_frozenset(row) for row in adjacency_matrix]
    min_reduction_neighbour: int = max(2, min(4, min_reduction_neighbour))
    max_reduction_neighbour: int = min(4, max(min_reduction_neighbour, max_reduction_neighbour))

    marked_nodes: List[int] = [node for node in range(len(roles)) if roles[node]]
    for node in marked_nodes:
        node_neighbours: FrozenSet[int] = nodes_neighbours[node]
        node_marked_neighbours: List[int] = [node for node in node_neighbours if roles[node]]

        for combination_num in range(min_reduction_neighbour, max_reduction_neighbour + 1):
            if len(node_marked_neighbours) < combination_num:
                break
            break_loop = False
            for neighbours_combination in itertools.combinations(node_marked_neighbours, combination_num):
                if not are_neighbours(neighbours_combination, nodes_neighbours):
                    continue
                all_neighbours = frozenset()
                for neighbour_node in neighbours_combination:
                    all_neighbours = all_neighbours.union(nodes_neighbours[neighbour_node])

                if not node_neighbours.issubset(all_neighbours):
                    continue

                roles[node] = 0
                break_loop = True
                break
            if break_loop:
                break


def are_neighbours(nodes: tuple, nodes_neighbours: List[FrozenSet[int]]):
    n = len(nodes)
    for comb_i in range(n):
        n1 = nodes[comb_i]
        n1_neighbours = nodes_neighbours[n1]
        for comb_j in range(comb_i + 1, n):
            n2 = nodes[comb_j]
            if n2 not in n1_neighbours:
                return False
    return True


def neighbours_array(adjacency_row: np.ndarray) -> np.ndarray:
    return np.where(adjacency_row)[0]


def neighbours_set(adjacency_row: np.ndarray) -> set:
    return set(neighbours_array(adjacency_row))


def neighbours_frozenset(adjacency_row: np.ndarray) -> frozenset:
    return frozenset(neighbours_array(adjacency_row))
