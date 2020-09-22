import numpy as np


def dominator(adjacency_matrix):
    roles = np.zeros((len(adjacency_matrix),), dtype='i8')
    dominator_construct(adjacency_matrix, roles)
    connected_pairs = dominator_connect(adjacency_matrix, roles)
    dominator_second_connect(adjacency_matrix, roles, connected_pairs)
    return roles


def dominator_construct(adjacency_matrix, roles):
    mask = np.zeros((len(adjacency_matrix),), dtype=np.uint8)
    # add fist producing node
    roles[0] = 1
    mask[adjacency_matrix[0] == 1] = 1
    mask[0] = 1

    indexes = np.arange(1, len(roles))
    np.random.shuffle(indexes)
    for i in indexes:
        if mask[i]:
            continue
        roles[i] = 1
        mask[adjacency_matrix[i] == 1] = 1
        mask[i] = 1


def get_first_common_neighbour(arr1, arr2):
    for i in range(len(arr1)):
        if not arr1[i]:
            continue
        if arr2[i]:
            return i
    return -1


def dominator_connect(adjacency, roles):
    connected_pairs = set()

    black_nodes = np.where(roles)[0]

    white_mask = ~roles

    for i, n_i in enumerate(black_nodes):
        for j in range(i + 1, len(black_nodes)):
            n_j = black_nodes[j]
            pair = (n_i, n_j)
            if pair in connected_pairs:
                continue
            i_white = adjacency[n_i] & white_mask
            j_white = adjacency[n_j] & white_mask

            common_i = get_first_common_neighbour(i_white, j_white)

            if common_i == -1:
                continue
            roles[common_i] = 1
            connected_pairs.add(pair)

    return connected_pairs


def get_unconnected_neighbours(i_white, j_white):
    i_w = set(np.where(i_white)[0])
    j_w = set(np.where(j_white)[0])

    for n_i in i_w:
        if n_i in j_w:
            continue
        for n_j in j_w:
            if n_j not in i_w:
                return min(n_i, n_j), max(n_i, n_j)
    return -1


def dominator_second_connect(adjacency, roles, connected_pairs):
    black_nodes = np.where(roles)[0]

    white_mask = ~roles

    for i, n_i in enumerate(black_nodes):
        for j in range(i + 1, len(black_nodes)):
            n_j = black_nodes[j]
            pair = (n_i, n_j)
            if pair in connected_pairs:
                continue
            i_white = adjacency[n_i] & white_mask
            j_white = adjacency[n_j] & white_mask

            common_pair = get_unconnected_neighbours(i_white, j_white)

            if common_pair == -1:
                continue
            roles[common_pair[0]] = roles[common_pair[1]] = 1
            connected_pairs.add(pair)
