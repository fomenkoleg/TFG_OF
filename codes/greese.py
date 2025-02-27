import numpy as np
from pprint import pprint
import random
import onmi
import dotenv

GROUND_TRUTH = env()

def data_collect():
    # Data recollection
    with open("datasets/email/email-main.txt", "r") as dataset_graph:
        #data = np.loadtxt(dataset_graph, dtype=int, max_rows=50)
        data = np.loadtxt(dataset_graph, dtype=int)

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
        # data = np.loadtxt(dataset_labels, dtype=int, max_rows=70)
        data = np.loadtxt(dataset_labels, dtype=int)


    unique_vertices = np.unique(data[:, 0])
    ground_truth = []
    for v in unique_vertices:
        groups = []
        for line in data:
            if int(line[0]) == v and len(line) > 1:
                groups.append(int(line[1]))
        ground_truth.append((int(v), *groups))
    return ground_truth
    

def first_arrangement(v_set):
    # For communities to overlap, there should be a minimum of 2 communities present
    n_communities = random.randint(2, round(len(v_set)/2))   
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
    
    return vertex_community_tuples

if __name__ == "__main__":
    e_set, e_count, v_set, v_count, adj_matrix = data_collect()

    # Grouping prediction
    prediction = first_arrangement(v_set)
    # print("Predicted groupings: ")
    # pprint(prediction)

    # Ground truth collection
    ground_truth = data_collect_groups()
    # print("Ground Truth Array: ")
    # pprint(ground_truth)

    # Efficiency comparison
    # Using ONMI
    accuracy = onmi.onmi(ground_truth, prediction)
    print(accuracy)
