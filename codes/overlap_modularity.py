import networkx as nx
import numpy as np
from collections import defaultdict

def overlapping_modularity(graph, node_communities):
    m = graph.number_of_edges()
    degree = dict(graph.degree())
    
    # Assign each node to its communities
    node_to_comms = defaultdict(set)
    for entry in node_communities:
        node = entry[0]
        for community in entry[1:]:
            node_to_comms[node].add(community)
    
    # Compute EQ using numpy for optimization
    eq = 0.0
    communities = set(comm for comms in node_to_comms.values() for comm in comms)
    for c in communities:
        edges = np.array([(i, j) for i, j in graph.edges() if c in node_to_comms[i] and c in node_to_comms[j]])
        if edges.size == 0:
            continue
        
        i_vals, j_vals = edges[:, 0], edges[:, 1]
        A_ij = np.ones(len(edges))
        k_i = np.array([degree[i] for i in i_vals])
        k_j = np.array([degree[j] for j in j_vals])
        O_i = np.array([len(node_to_comms[i]) for i in i_vals])
        O_j = np.array([len(node_to_comms[j]) for j in j_vals])
        
        eq += np.sum((A_ij - (k_i * k_j) / (2 * m)) * (1 / (O_i * O_j)))
    
    return eq / (2 * m)
