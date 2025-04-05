import numpy as np

def parse_cluster_data(data):
    """Parses clustering data into a binary matrix."""
    nodes = sorted(set(node for node, *groups in data))
    communities = sorted(set(comm for _, *groups in data for comm in groups))
    
    node_index = {node: i for i, node in enumerate(nodes)}
    comm_index = {comm: i for i, comm in enumerate(communities)}
    
    matrix = np.zeros((len(nodes), len(communities)), dtype=int)
    for node, *groups in data:
        for group in groups:
            matrix[node_index[node], comm_index[group]] = 1
    return matrix

def entropy(X):
    """Computes entropy H(X)."""
    n = X.shape[0]
    node_degrees = X.sum(axis=1)  # Number of communities per node
    total_degrees = np.sum(node_degrees)
    if total_degrees == 0:
        return 0.0
    
    p = node_degrees / total_degrees  # Probability distribution
    p = p[p > 0]  # Remove zero probabilities
    return -np.sum(p * np.log2(p))  # Handle empty case

def joint_entropy(X, Y):
    """Computes joint entropy H(X, Y)."""
    xy = X.T @ Y
    total = np.sum(xy)
    if total == 0:  # Handle edge case when matrices have no overlap
        return 0.0
    p_xy = xy / total
    p_xy = p_xy[p_xy > 0]  # Remove zero probabilities
    return -np.sum(p_xy * np.log2(p_xy)) if len(p_xy) > 0 else 0.0  # Handle empty case

def mutual_information(X, Y):
    """Compute mutual information between two clusterings.
    Args:
        X, Y: Binary matrices (nodes × communities).
    Returns:
        I: Mutual information (float), robust to sparse overlaps.
    """
    n_nodes = X.shape[0]
    
    # Joint probability P(X, Y) = (X.T @ Y) / n_nodes
    joint = (X.T @ Y) / n_nodes
    # joint = joint[joint > 0]  # Ignore zero-probability terms
    
    # Marginal probabilities P(X) and P(Y)
    p_x = np.sum(X, axis=0) / n_nodes
    p_y = np.sum(Y, axis=0) / n_nodes
    p_x = p_x[p_x > 0]
    p_y = p_y[p_y > 0]
    
    # MI = Σ P(X,Y) log(P(X,Y) / (P(X)P(Y)))
    # Vectorized computation for efficiency
    mi = np.sum(joint * np.log2(joint / (p_x[:, None] * p_y[None, :])))
    return max(mi, 0)

def onmi(gt, pred):
    """Computes ONMI based on the given formula."""
    X = parse_cluster_data(gt)
    Y = parse_cluster_data(pred)
    
    I_XY = mutual_information(X, Y)
    H_X = entropy(X)
    H_Y = entropy(Y)
    
    # Handle edge cases
    if H_X == 0 and H_Y == 0:  # Both empty or single cluster
        return 1.0 if np.array_equal(X, Y) else 0.0
    elif H_X == 0 or H_Y == 0:  # One clustering has zero entropy
        return 0.0
    return 2*I_XY / (H_X + H_Y)  # Avoid division by zero