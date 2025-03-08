import numpy as np
import copy
from pprint import pprint
from collections import Counter

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

def assign_comms_recursively(adj_matrix, n, com_act, communities_per_node, group_depth, visited):
    
    if group_depth > 0:
        neighbors = np.where(adj_matrix[n] == 1)[0]

        for ne in neighbors:
            if not visited[ne]:
                visited[ne] = True
                communities_per_node[ne].append(com_act)
                return assign_comms_recursively(adj_matrix, ne, com_act, communities_per_node, group_depth-1, visited)
    
    return communities_per_node, visited

def overlap_communities(adj_matrix, v_set, v_count, percent_sim, communities_per_node):

    overlapped_communities = copy.deepcopy(communities_per_node)

    for node, _ in communities_per_node:
        # get neigbors of each node
        # if a significant percentage of the neighbors belong 
        # to a different group from the orignal, add new 
        # community to the solution

        neighbors = np.where(adj_matrix[node] == 1)[0]
        neighbor_comms = [communities_per_node[n][1] for n in neighbors]

        comm_count = Counter(neighbor_comms)

        for comm in list(map(int, np.unique(neighbor_comms))):
            if overlapped_communities[node]:
                if comm not in overlapped_communities[node]:
                    # if the actual node belongs to more communities, add the community to the node's array
                    if comm_count[comm] / len(neighbors) >= percent_sim:
                        overlapped_communities[node].append(comm)

    return overlapped_communities
            

def greedysol(adj_matrix, v_set, v_count, percent_sim, group_depth = 2):    
    cands = get_candidates(adj_matrix, v_set)        

    # communities_per_node defined as array of tuples where each node has 1 or more communities
    communities_per_node = [[i] for i in range(v_count)]
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

                communities_per_node, visited = assign_comms_recursively(adj_matrix, n, com_act, communities_per_node, group_depth, visited)

            com_act += 1

    return overlap_communities(adj_matrix, v_set, v_count, percent_sim, communities_per_node)