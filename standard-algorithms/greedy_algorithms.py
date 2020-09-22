import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def greedy_connect_without_prune(adjacency_matrix):
    roles = np.zeros((len(adjacency_matrix),), dtype='i8')
    greedy_construct(adjacency_matrix, roles)
    return roles


def greedy_connect(adjacency_matrix):
    roles = greedy_connect_without_prune(adjacency_matrix)
    greedy_prune(adjacency_matrix, roles)
    return roles


def greedy_construct(adjacency_matrix, roles):
    n = len(adjacency_matrix)
    node_mask = np.ones((n,), dtype='i8')
    dominated = set()

    # index_with_big_degree = np.argmax(
    #     np.sum(
    #         adjacency_matrix * node_mask,
    #         axis=1
    #     )
    # )
    index_with_big_degree = 0
    roles[index_with_big_degree] = 1
    neighbours_mask = adjacency_matrix[index_with_big_degree] == 1
    node_mask[neighbours_mask] = 0
    node_mask[index_with_big_degree] = 0

    dominated.update(np.where(neighbours_mask)[0])

    for i in range(n - 1):
        max_index, white_heighbours_indexes = argmax_dominated(dominated, adjacency_matrix, node_mask)

        if (max_index == -1): break

        roles[max_index] = 1
        node_mask[white_heighbours_indexes] = 0
        node_mask[max_index] = 0
        dominated.update(white_heighbours_indexes)
        dominated.remove(max_index)


def greedy_prune(adjacency_matrix, roles):
    for i in range(1, len(roles)):
        if not roles[i]: continue

        # check if all grey neighbours have black neighbours
        suspicious = True
        neighbours = np.where(adjacency_matrix[i])[0]
        for n_i in neighbours:
            if not roles[n_i] and not has_black_neighbours(i, adjacency_matrix[n_i], roles):
                suspicious = False
                break

        if not suspicious: continue

        # check if after removing node the dominating set still be connected
        black_neighbours = [n_i for n_i in neighbours if roles[n_i]]
        for n_i in black_neighbours:
            if not has_path_to_nodes(i, n_i, black_neighbours, adjacency_matrix, roles):
                suspicious = False
                break

        if suspicious: roles[i] = 0


# --- Helper function ---

def has_black_neighbours(parent_node, neighbours, roles):
    for i, neighbour in enumerate(neighbours):
        if i != parent_node and neighbour and roles[i]:
            return True
    return False


def argmax_dominated(dominated, adjacency_matrix, node_mask):
    max_index, max_degree = -1, 0
    white_neighbours_indexes = None
    for index in dominated:
        white_neighbours = adjacency_matrix[index] * node_mask
        node_degree = np.sum(white_neighbours)
        if node_degree > max_degree:
            max_index = index
            max_degree = node_degree
            white_neighbours_indexes = np.where(white_neighbours)[0]
    return max_index, white_neighbours_indexes


def has_path_to_node(current_node, start_node, end_node, adjacency_matrix, roles):
    # Mark all the vertices as not visited
    visited = [False] * len(adjacency_matrix)
    visited[start_node] = True

    queue = [start_node]

    while queue:
        n_i = queue.pop(0)

        if n_i == end_node:
            return True

        #  Else, continue to do BFS
        for i in adjacency_matrix[n_i]:
            # add only relay nodes and pass current_node
            if i and roles[i] and i != current_node and not visited[i]:
                queue.append(i)
                visited[i] = True

    # If BFS is complete without visited end_node
    return False


def has_path_to_nodes(current_node, start_node, end_nodes, adjacency_matrix, roles):
    for end_node in end_nodes:
        if (end_node != start_node and
                not has_path_to_node(current_node, start_node, end_node, adjacency_matrix, roles)):
            return False
    return True


# --- Testing functions ---

def show_graph_with_labels(adjacency, roles, positions, only_relays=False, save_to=None):
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

def test_greedy_construct():
    roles = np.zeros((len(adjacency_matrix),), dtype='i8')
    greedy_construct(adjacency_matrix, roles)
    np.testing.assert_array_equal(
        #    0  1  2  3  4  5  6  7  8  9 10 11 12
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        roles
    )


def test_greedy_prune():
    roles = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    greedy_prune(adjacency_matrix, roles)
    np.testing.assert_array_equal(
        #    0  1  2  3  4  5  6  7  8  9 10 11 12
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        roles
    )


if __name__ == "__main__":
    adjacency_matrix = np.array([
        #    0  1  2  3  4  5  6  7  8  9 10 11 12
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    ])

    test_greedy_construct()
    test_greedy_prune()

    roles = greedy_connect(adjacency_matrix)

    print(roles)

    show_graph_with_labels(adjacency_matrix, roles)
