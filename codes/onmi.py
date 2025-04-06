import numpy as np

def parse_cluster_data(data):
    """Parses clustering data into a binary matrix."""
    nodes = sorted(set(node for node, *_ in data))
    communities = sorted(set(comm for _, *groups in data for comm in groups))
    
    node_index = {node: i for i, node in enumerate(nodes)}
    comm_index = {comm: i for i, comm in enumerate(communities)}
    
    matrix = np.zeros((len(nodes), len(communities)), dtype=int)
    for node, *groups in data:
        for group in groups:
            matrix[node_index[node], comm_index[group]] = 1
    return matrix

def onmi(pred, gt):

    X = parse_cluster_data(pred)
    Y = parse_cluster_data(gt)

    n_nodes = X.shape[0]
    eps = np.finfo(float).eps  # Small constant to avoid log(0)
    
    # Entropy H(X)
    p_X = np.sum(X, axis=0) / n_nodes  # P(community k in X)
    H_X = -np.sum(p_X * np.log(p_X + eps))
    
    # Entropy H(Y)
    p_Y = np.sum(Y, axis=0) / n_nodes  # P(community l in Y)
    H_Y = -np.sum(p_Y * np.log(p_Y + eps))
    
    # Mutual Information I(X, Y)
    I_XY = 0.0
    for k in range(X.shape[1]):  # For each community in X
        for l in range(Y.shape[1]):  # For each community in Y
            # Joint probability P(node in community k AND l)
            p_joint = np.sum(X[:, k] * Y[:, l]) / n_nodes
            p_k = p_X[k]  # P(community k in X)
            p_l = p_Y[l]  # P(community l in Y)
            I_XY += p_joint * np.log2((p_joint + eps) / (p_k * p_l + eps))
    
    # Normalization: I(X,Y) / max(H(X), H(Y))
    
    
    ONMI = I_XY / max(H_X, H_Y)
    return ONMI  # Ensures result is in [0, 1