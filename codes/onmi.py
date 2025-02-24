import numpy as np

def compute_entropy(cluster_matrix):
    """Computes the entropy of a given cluster matrix."""
    n, k = cluster_matrix.shape
    probabilities = cluster_matrix.sum(axis=0) / n
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
    return -np.sum(probabilities * np.log2(probabilities))

def compute_mutual_information(X, Y):
    """Computes mutual information between two clustering matrices."""
    n, kx = X.shape
    _, ky = Y.shape
    
    mi = 0
    for i in range(kx):
        for j in range(ky):
            a = np.sum((X[:, i] == 1) & (Y[:, j] == 1))
            b = np.sum((X[:, i] == 0) & (Y[:, j] == 1))
            c = np.sum((X[:, i] == 1) & (Y[:, j] == 0))
            d = np.sum((X[:, i] == 0) & (Y[:, j] == 0))

            if a + d < b + c:  # Fix unintuitive behavior
                conditional_entropy = (b + d) / n * np.log2((b + d) / n) + (a + c) / n * np.log2((a + c) / n)
            else:
                conditional_entropy = (a / n) * np.log2(a / n) + (d / n) * np.log2(d / n)

            mi += conditional_entropy

    return mi

def normalized_mutual_information(ground_truth, predicted):
    """Computes Normalized Mutual Information (NMI) with proper normalization."""
    nodes = list(set(node for node, *groups in ground_truth) | set(node for node, *groups in predicted))
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Build cluster matrices
    num_nodes = len(nodes)
    gt_communities = list(set(c for _, *groups in ground_truth for c in groups))
    pred_communities = list(set(c for _, *groups in predicted for c in groups))
    
    X = np.zeros((num_nodes, len(gt_communities)), dtype=int)
    Y = np.zeros((num_nodes, len(pred_communities)), dtype=int)

    for node, *groups in ground_truth:
        for group in groups:
            X[node_index[node], gt_communities.index(group)] = 1
    
    for node, *groups in predicted:
        for group in groups:
            Y[node_index[node], pred_communities.index(group)] = 1

    # Compute entropies and mutual information
    H_X = compute_entropy(X)
    H_Y = compute_entropy(Y)
    I_XY = compute_mutual_information(X, Y)

    # Normalize using max entropy
    NMI = I_XY / max(H_X, H_Y)
    
    return NMI