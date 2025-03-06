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
    p_x = np.sum(X, axis=0) / n  # Probability of each cluster
    p_x = p_x[p_x > 0]  # Remove zero probabilities
    return -np.sum(p_x * np.log2(p_x)) if len(p_x) > 0 else 0.0  # Handle empty case

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
    """Computes mutual information I(X : Y)."""
    return entropy(X) + entropy(Y) - joint_entropy(X, Y)

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
    return I_XY / max(H_X, H_Y, 1e-10)  # Avoid division by zero
