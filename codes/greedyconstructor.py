import numpy as np
from pprint import pprint

def get_candidates(adj_matrix, v_set):
    cands = []
    # For each node in the set of nodes, set its corresponding 
    # candidate as a tuple of (node, neighbornumber)
    for node in v_set:
        neighbor_number = np.sum(adj_matrix[node])
        cands.append([node, neighbor_number])

    # Sort the candidates by the number of neighbors
    cands.sort(key=lambda x: x[1], reverse=True)
    print("Candidates array")
    print(len(cands))
    return cands

def greedysol(adj_matrix, v_set, e_set, e_count, v_count, percent_sim):    
    cands = get_candidates(adj_matrix, v_set)        

    # communities_per_node defined as array of tuples where each node has 1 or more communities
    communities_per_node = [[] for _ in range(v_count)]
    com_act = 0
    visited =  [False for _ in range(v_count)]

    # For each candidate, add its neighbors to its community
    for cand in cands:
        # Get the best suitable node and attach a community and its neighbors
        node = cand.pop(0)
        if not visited[node]:
            visited[node] = True
            communities_per_node[node].append(com_act)
            neighbors = np.where(adj_matrix[node] == 1)[0]

            # For each neighbor, assign the community of the original node
            for n in neighbors:
                if not visited[n]:
                    communities_per_node[n].append(com_act)
                    visited[n] = True
            com_act += 1

    return communities_per_node          

    