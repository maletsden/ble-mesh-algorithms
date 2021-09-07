import unittest
import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append("../../standard-algorithms")

import k2_pruning


def show_graph_with_labels(adjacency, roles, positions=None, only_relays=False, save_to=None):
    rows, cols = np.where(adjacency == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    gr = nx.Graph()

    nodes = range(adjacency.shape[0])
    gr.add_nodes_from(nodes)

    pos = positions or nx.spring_layout(gr)
    relays = set(np.where(roles)[0])
    nx.draw_networkx_nodes(gr, pos,
                           nodelist=relays,
                           node_color='black',
                           node_size=300)

    nx.draw_networkx_edges(gr, pos,
                           edgelist=[edge for edge in edges if edge[0] in relays and edge[1] in relays],
                           width=1, edge_color='black')

    if not only_relays:
        nx.draw_networkx_edges(gr, pos,
                               edgelist=edges,
                               width=1, edge_color='grey')
        nx.draw_networkx_nodes(gr, pos,
                               nodelist=np.where([not role for role in roles])[0],
                               node_color='grey',
                               node_size=300)
        nx.draw_networkx_labels(gr, pos, {node: node for node in nodes},
                                font_size=14, font_color="white")
    else:
        nx.draw_networkx_labels(gr, pos, {node: node for node in relays},
                                font_size=14, font_color="white")

    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


class TestK2Pruning(unittest.TestCase):
    def test_k2_pruning(self):
        adjacency_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        expected_roles = np.array([0, 1, 0, 0, 0], dtype='i8')

        roles = k2_pruning.k2_pruning(adjacency_matrix)
        show_graph_with_labels(adjacency_matrix, roles)

        np.testing.assert_array_equal(expected_roles, roles)

    def test_construct_stage(self):
        adjacency_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        expected_roles = np.array([1, 1, 0, 0, 0], dtype='i8')
        roles = np.zeros_like(expected_roles)

        k2_pruning.k2_pruning_construct(adjacency_matrix, roles)
        show_graph_with_labels(adjacency_matrix, roles)

        np.testing.assert_array_equal(expected_roles, roles)

    def test_first_reduction_stage(self):
        adjacency_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        expected_roles = np.array([0, 1, 0, 0, 0], dtype='i8')
        roles = np.array([1, 1, 1, 0, 0], dtype='i8')

        k2_pruning.k2_pruning_first_reduction(adjacency_matrix, roles)
        show_graph_with_labels(adjacency_matrix, roles)

        np.testing.assert_array_equal(expected_roles, roles)

    def test_second_reduction_stage(self):
        adjacency_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        expected_roles = np.array([0, 1, 1, 0, 0], dtype='i8')
        roles = np.array([1, 1, 1, 0, 0], dtype='i8')

        k2_pruning.k2_pruning_second_reduction(adjacency_matrix, roles)
        show_graph_with_labels(adjacency_matrix, roles)

        np.testing.assert_array_equal(expected_roles, roles)

    def test_third_reduction_stage(self):
        adjacency_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        expected_roles = np.array([0, 1, 1, 1, 0], dtype='i8')
        roles = np.array([1, 1, 1, 1, 0], dtype='i8')

        k2_pruning.k2_pruning_second_reduction(adjacency_matrix, roles,
                                               min_reduction_neighbour=3, max_reduction_neighbour=3)
        show_graph_with_labels(adjacency_matrix, roles)

        np.testing.assert_array_equal(expected_roles, roles)


if __name__ == '__main__':
    unittest.main()
