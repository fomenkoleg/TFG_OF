import numpy as np
from pprint import pprint
import random

def data_collect():
    # Data recollection
    with open("datasets/email/email-main.txt", "r") as dataset_graph:
        data = np.loadtxt(dataset_graph, dtype=int, max_rows=50)

    # Extract unique vertices and count them
    e_set = set(map(tuple, data))
    v_set = np.unique(data)
    v_count = len(v_set)
    e_count = len(e_set)

    # Create adjacency matrix using numpy indexing
    adj_matrix = np.zeros((v_count, v_count), dtype=int)
    adj_matrix[data[:, 0], data[:, 1]] = 1

    # Graph representation
    print("Graph representation")
    print("Vertex set length:", v_count)
    print("Edge set length:", e_count)

    return e_set, e_count, v_set, v_count, adj_matrix

def similarity(v1, v2, adj_matrix):
    neighbors_v1 = set(neighbors(v1, adj_matrix))
    neighbors_v2 = set(neighbors(v2, adj_matrix))
    return len(neighbors_v1 & neighbors_v2)

def neighbors(v, adj_matrix):
    return np.where(adj_matrix[v] == 1)[0]

def f_funct(v, adj_matrix, s):
    neighbors_v = set(neighbors(v, adj_matrix))
    return len(neighbors_v & s) / len(s) 

def data_collect_groups():
    with open("datasets/email/email-labels.txt", "r") as dataset_labels:
        data = np.loadtxt(dataset_labels, dtype=int, max_rows=70)

    unique_vertices = np.unique(data[:, 0])
    return [tuple([int(v)] + list(map(int, data[data[:, 0] == v, 1]))) for v in unique_vertices]

def first_arrangement(v_set, e_set, adj_matrix):

    n_communities = random.randint(4, round(len(v_set)/10))   
    communities = list(range(n_communities))

    communities_per_v = np.random.choice(communities, len(v_set))
    more_than_one_comm_percent = 0.2
    more_than_one_comm_nodes = v_set[np.random.rand(len(v_set)) < more_than_one_comm_percent]
    vertex_community_tuples = []
    for v in v_set:
        if v in more_than_one_comm_nodes:
            # Define random number of extra communities
            extra_comms = random.randint(1, 3)
            extra_comm_array = [int(communities_per_v[v])]
            for e in range(extra_comms):
                comm = -1
                while comm == -1 or int(comm) == int(communities_per_v[v]) or comm in extra_comm_array:
                    comm = np.random.choice(communities)
                extra_comm_array.append(int(comm))
                
            t = (int(v), *sorted(extra_comm_array))
            vertex_community_tuples.append(t)
        else:
            t = (int(v), int(communities_per_v[v]))
            vertex_community_tuples.append(t)
    
    pprint(vertex_community_tuples)

if __name__ == "__main__":
    e_set, e_count, v_set, v_count, adj_matrix = data_collect()
    first_arrangement(v_set, e_set, adj_matrix)
    groups = data_collect_groups()
    pprint(groups)