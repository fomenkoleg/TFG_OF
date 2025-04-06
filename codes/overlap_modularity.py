import numpy as np
from collections import defaultdict

def parse_cluster_data(cluster_data):
    """Parse clustering data into efficient data structures.
    
    Args:
        cluster_data: List of lists where each sublist is [node, comm1, comm2, ...]
    
    Returns:
        tuple: (community_matrix, node_to_comms, comm_to_nodes)
    """
    nodes = sorted({item[0] for item in cluster_data})
    communities = sorted({comm for item in cluster_data for comm in item[1:]})
    
    node_idx = {node: i for i, node in enumerate(nodes)}
    comm_idx = {comm: i for i, comm in enumerate(communities)}
    
    # Binary matrix (nodes x communities)
    community_matrix = np.zeros((len(nodes), len(communities)), dtype=int)
    
    # Mapping structures
    node_to_comms = defaultdict(set)
    comm_to_nodes = defaultdict(set)
    
    for item in cluster_data:
        node = item[0]
        for comm in item[1:]:
            community_matrix[node_idx[node], comm_idx[comm]] = 1
            node_to_comms[node].add(comm)
            comm_to_nodes[comm].add(node)
    
    return community_matrix, node_to_comms, comm_to_nodes

def compute_degrees(edges):
    """Compute degree for each node from edge list."""
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return degree

def overlapping_modularity(edges, cluster_data):
    """Compute overlapping modularity for a given clustering.
    
    Args:
        edges: List of tuples representing graph edges
        cluster_data: List of lists where each sublist is [node, comm1, comm2, ...]
    
    Returns:
        float: Overlapping modularity score
    """
    if not edges or not cluster_data:
        return 0.0
    
    # Preprocess data
    e_count = len(edges)
    degree = compute_degrees(edges)
    _, node_to_comms, comm_to_nodes = parse_cluster_data(cluster_data)
    
    total_modularity = 0.0
    
    for comm in comm_to_nodes:
        # Find all node pairs in this community that have edges
        comm_nodes = comm_to_nodes[comm]
        comm_edges = [(u, v) for u, v in edges 
                      if u in comm_nodes and v in comm_nodes]
        
        if not comm_edges:
            continue
        
        # Vectorized calculations
        u_nodes, v_nodes = zip(*comm_edges)
        A_ij = 1  # Since we filtered existing edges
        
        # Get degrees and overlaps for all nodes in these edges
        k_i = np.array([degree[u] for u in u_nodes])
        k_j = np.array([degree[v] for v in v_nodes])
        O_i = np.array([len(node_to_comms[u]) for u in u_nodes])
        O_j = np.array([len(node_to_comms[v]) for v in v_nodes])
        
        # Compute modularity contribution for this community
        community_contribution = np.sum(
            (A_ij - (k_i * k_j) / (2 * e_count)) / (O_i * O_j))
        
        total_modularity += community_contribution
    
    return total_modularity / (2 * e_count)