import numpy as np
import copy
from collections import Counter

def get_candidates(adj_matrix, v_set):
    """Get candidate nodes sorted by the number of neighbors."""
    cands = []
    for node in v_set:
        neighbor_number = np.sum(adj_matrix[node])
        cands.append([node, neighbor_number])
    cands.sort(key=lambda x: x[1], reverse=True)  # Sort by number of neighbors
    return cands

def assign_comms_recursively(adj_matrix, node, com_act, communities_per_node, depth, visited):
    """
    Recursively assign communities to neighbors up to a specified depth.
    """
    if depth > 0:
        neighbors = np.where(adj_matrix[node] == 1)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                communities_per_node[neighbor].append(com_act)
                # Recursively assign communities to the neighbor's neighbors
                assign_comms_recursively(adj_matrix, neighbor, com_act, communities_per_node, depth - 1, visited)
    return communities_per_node, visited

def overlap_communities(adj_matrix, v_set, v_count, percent_sim, communities_per_node):
    """
    Overlap communities based on neighbor similarity.
    """
    overlapped_communities = copy.deepcopy(communities_per_node)
    for node in range(v_count):
        neighbors = np.where(adj_matrix[node] == 1)[0]
        neighbor_comms = [communities_per_node[n][1] for n in neighbors if len(communities_per_node[n]) > 1]
        comm_count = Counter(neighbor_comms)
        for comm in comm_count:
            if comm not in overlapped_communities[node]:
                if comm_count[comm] / len(neighbors) >= percent_sim:
                    overlapped_communities[node].append(comm)
    return overlapped_communities

def greedysol(adj_matrix, v_set, v_count, percent_sim, group_depth=2):
    """
    Greedy algorithm to detect overlapping communities.
    """
    cands = get_candidates(adj_matrix, v_set)
    communities_per_node = [[i] for i in range(v_count)]  # Each node initially belongs to its own community
    com_act = 0  # Current community ID
    visited = [False for _ in range(v_count)]  # Track visited nodes

    for cand in cands:
        node = cand[0]
        if not visited[node]:
            visited[node] = True
            communities_per_node[node].append(com_act)
            # Assign the community to neighbors recursively
            communities_per_node, visited = assign_comms_recursively(
                adj_matrix, node, com_act, communities_per_node, group_depth, visited
            )
            com_act += 1  # Move to the next community

    return overlap_communities(adj_matrix, v_set, v_count, percent_sim, communities_per_node)