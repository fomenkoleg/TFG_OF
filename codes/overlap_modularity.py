import numpy as np
from collections import defaultdict

def parse_cluster_data(data):
    """Parses clustering data into a binary matrix and community mappings."""
    nodes = sorted(set(node for node, *groups in data))
    communities = sorted(set(comm for _, *groups in data for comm in groups))
    
    node_index = {node: i for i, node in enumerate(nodes)}
    comm_index = {comm: i for i, comm in enumerate(communities)}
    
    matrix = np.zeros((len(nodes), len(communities)), dtype=int)
    node_to_comms = defaultdict(set)
    
    for node, *groups in data:
        for group in groups:
            matrix[node_index[node], comm_index[group]] = 1
            node_to_comms[node].add(group)
    
    return matrix, node_to_comms

def overlapping_modularity(edges, e_count, node_communities):
    """Computes the overlapping modularity of a given clustering."""
    # Calculate degree for each node
    degree = defaultdict(int)
    for edge in edges:
        # This is for undirected graphs
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    
    # Parse the cluster data
    _, node_to_comms = parse_cluster_data(node_communities)
    eq = 0.0
    
    # Get all unique communities
    communities = set(comm for comms in node_to_comms.values() for comm in comms)
    
    for c in communities:
        # Find edges where both nodes belong to the community c
        filtered_edges = [(i, j) for i, j in edges if c in node_to_comms[i] and c in node_to_comms[j]]
        
        if len(filtered_edges) == 0:
            continue
        
        # Compute required arrays
        i_vals, j_vals = zip(*filtered_edges)
        A_ij = np.ones(len(filtered_edges))
        k_i = np.array([degree[i] for i in i_vals])
        k_j = np.array([degree[j] for j in j_vals])
        O_i = np.array([len(node_to_comms[i]) for i in i_vals])
        O_j = np.array([len(node_to_comms[j]) for j in j_vals])
        
        eq += np.sum((A_ij - (k_i * k_j) / (2 * e_count)) * (1 / (O_i * O_j)))
    
    return eq / (2 * e_count) if e_count > 0 else 0
