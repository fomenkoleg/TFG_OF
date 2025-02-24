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

def h(w, n):
    """Entropy contribution: h(w, n) = -w * log2(w/n), with check to avoid log2(0)."""
    if w == 0:
        return 0
    value = -w * np.log2(w / n)
    return max(value, 0)  # Ensure no negative contribution

def conditional_entropy(Xi, Yj):
    """Conditional entropy H*(Xi | Yj) from Eq. (2) of the paper."""
    n = len(Xi)
    a = np.sum((Xi == 1) & (Yj == 1))
    b = np.sum((Xi == 0) & (Yj == 1))
    c = np.sum((Xi == 1) & (Yj == 0))
    d = np.sum((Xi == 0) & (Yj == 0))
    
    # Calculate entropies
    h_a, h_b, h_c, h_d = h(a, n), h(b, n), h(c, n), h(d, n)
    h_bd, h_ac = h(b + d, n), h(a + c, n)
    h_cd, h_ab = h(c + d, n), h(a + b, n)
    
    # Conditional entropy calculation
    H_Xi_Yj = h_a + h_b + h_c + h_d - h_bd - h_ac
    H_Xi_Yj = max(H_Xi_Yj, 0)  # Ensure no negative values
    
    # Apply technical correction (if vectors are near complements)
    return H_Xi_Yj if h_a + h_d >= h_b + h_c else h_cd + h_ab

def H_given(X, Y):
    """H(X | Y): sum of minimal conditional entropies."""
    return np.sum([min(conditional_entropy(X[:, i], Y[:, j]) for j in range(Y.shape[1])) 
                   for i in range(X.shape[1])])

def entropy(X):
    """Entropy H(X) from the paper."""
    n, k = X.shape
    return np.sum([h(np.sum(X[:, i] == 1), n) + h(np.sum(X[:, i] == 0), n) for i in range(k)])

def mutual_information(X, Y):
    """Mutual information I(X : Y) from Eq. (5)."""
    H_X, H_Y = entropy(X), entropy(Y)
    H_X_given_Y = H_given(X, Y)
    H_Y_given_X = H_given(Y, X)
    I_XY = 0.5 * (H_X - H_X_given_Y + H_Y - H_Y_given_X)
    return max(I_XY, 0)  # Ensure mutual information is non-negative

def nmi_max(X, Y):
    """NMI using max entropy: I(X : Y) / max(H(X), H(Y))"""
    I_XY = mutual_information(X, Y)
    H_X, H_Y = entropy(X), entropy(Y)
    return I_XY / max(max(H_X, H_Y), 1e-10)

def onmi(gt, pred):
    X = parse_cluster_data(gt)
    Y = parse_cluster_data(pred)
    return nmi_max(X, Y)