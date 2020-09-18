import numpy as np
def dominator(adjacency_matrix):
    roles = np.zeros((len(adjacency_matrix),), dtype='i8')
    dominator_construct(adjacency_matrix, roles)
    dominator_connect(adjacency_matrix, roles)
    dominator_second_connect(adjacency_matrix, roles)
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
        if mask[i]: continue
        roles[i] = 1
        mask[adjacency_matrix[i] == 1] = 1
        mask[i] = 1

def dominator_connect(adjacency_matrix, roles):
    n = len(roles)
    connected_pairs = set()
    
    for i in range(n):
        if roles[i]: continue
        black_neighbours = np.where(adjacency_matrix[i] * roles)[0]
        connecting_node = False
        for b_i in range(len(black_neighbours)):
            for b_j in range(b_i + 1, len(black_neighbours)):
                if (b_i, b_j) not in connected_pairs:
                    connected_pairs.add((b_i, b_j))
                    connecting_node = True
        if connecting_node: roles[i] = 1

def dominator_second_connect(adjacency_matrix, roles):
    connected_pairs = set()
    
    n = len(roles)
    
    for i in range(n):
        # nodes must be dominated
        if roles[i]: continue
        
        u1_neighbours = adjacency_matrix[i]
        u1_black_neighbours = np.where(u1_neighbours * roles)[0]
        
        for j in range(i + 1, n):
            # nodes must be dominated
            if roles[j]: continue
            
            u2_neighbours = adjacency_matrix[j]
            u2_black_neighbours = np.where(u2_neighbours * roles)[0]
            
            connecting_nodes = False
            for u1_n_i in u1_black_neighbours:
                break_loop = False
                for u2_n_i in u2_black_neighbours:
                    # tricky part, probably this "if"(u1_n_i == u2_n_i) should break these 2 loops (not mentioned in artickle)
                    if u1_n_i == u2_n_i:
                        break_loop = True
                        break
                    pair = (min(u1_n_i, u2_n_i), max(u1_n_i, u2_n_i))
                    if u2_neighbours[u1_n_i] or u1_neighbours[u2_n_i] or \
                        pair in connected_pairs: continue
                    
                    connected_pairs.add(pair)
                    connecting_nodes = True
                if break_loop: break
            if connecting_nodes: roles[i] = roles[j] = 1
# dom_nodes = np.copy(con_roles)
                    
# dominator_second_connect(adjacency_matrix, dom_nodes)
# dom_nodes
# show_graph_with_labels(adjacency_matrix, dom_nodes, positions, True)
# np.count_nonzero(dom_nodes)    
# con_roles = np.copy(node_roles)
                    
# dominator_connect(adjacency_matrix, con_roles)
# con_roles
# show_graph_with_labels(adjacency_matrix, con_roles, positions, True)

# if __name__ == "__main__":
#     adjacency_matrix = np.array([
#         #    0  1  2  3  4  5  6  7  8  9 10 11 12
#         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#         [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
#         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
#         [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
#     ])

#     test_greedy_construct()
#     test_greedy_prune()

#     roles = greedy_connect(adjacency_matrix)

#     print(roles)

#     show_graph_with_labels(adjacency_matrix, roles)
