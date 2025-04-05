import numpy as np
from pprint import pprint
import random
import onmi
import overlap_modularity
import greedyconstructor
from time import sleep
from prettytable import PrettyTable


OVERLAPPING = True
GROUNDTRUTH = True

GRAPH_FILE = "datasets/email/email-main.txt"
GT_FILE = "datasets/email/email-labels.txt"

def data_collect():
    # Data recollection
    with open(GRAPH_FILE, "r") as dataset_graph:
        data = np.loadtxt(dataset_graph, dtype=int)
    # Extract unique vertices and count them
    e_set = set(map(tuple, data))
    v_set = np.unique(data)
    v_count = len(v_set)
    e_count = len(e_set)

    # Create adjacency matrix using numpy indexing

    # use max(v_set) to ensure that the matrix is square and that it has as many spaces as needed
    # ensure for non-consecutive nodes and non-zero indexed nodes
    adj_matrix = np.zeros((max(v_set)+1, max(v_set)+1), dtype=int)
    adj_matrix[data[:, 0], data[:, 1]] = 1
    adj_matrix[data[:, 1], data[:, 0]] = 1


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
    data = []
    with open(GT_FILE, "r") as file:
        for line in file:
            columns = line.strip().split()
            data.append(list(map(int, columns)))

    vertices = list(range(len(data)))
    ground_truth = []
    for v in vertices:
        groups = []
        for line in data:
            if v == line[0]:
                groups.append(*line[1:])
        ground_truth.append([v, *groups])
    return ground_truth

def first_arrangement(v_set):
    # For communities to overlap, there should be a minimum of 2 communities present
    n_communities = random.randint(2, round(len(v_set)/2))  
    communities = list(range(1, n_communities))

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

    #pprint(adj_matrix)
    # Grouping prediction
    # prediction = first_arrangement(v_set)
    # print("Predicted groupings: ")
    # pprint(prediction)

    # Ground truth collection
    ground_truth = data_collect_groups()
    #print("Ground Truth Array: ")
    #pprint(ground_truth)

    # # Testing numbers
    # for _ in range(10):
    #     # Grouping prediction
    #     prediction = first_arrangement(v_set)
    #     accuracy = onmi.onmi(ground_truth, prediction)
    #     print("ONMI: ", end="")
    #     print(accuracy)
    #     print("Overlapping Modularity: ", end="")
    #     accuracy = overlap_modularity.overlapping_modularity(e_set, e_count, prediction)
    #     print(accuracy) 

    # Grouping prediction

    accuracies = []
    # percentages = [0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    percentages = [0.8, 0.01]
    depth = 1
    for percentage in percentages:
        print()
        print(f"Calculating overlapping community prediction with {percentage*100}% of nodes belonging to other communities")
        prediction = greedyconstructor.greedysol(adj_matrix, v_set, v_count, percentage, depth)
        # print("Predicted groupings: ")
        #pprint(prediction)
        # Efficiency comparison
        # Using ONMI
        accuracy = onmi.onmi(ground_truth, prediction)
        # pprint(ground_truth[:20])
        # pprint(prediction[:20])
        
        print("ONMI:", accuracy)
        accuracies.append(accuracy)
        # Using Overlapping Modularity
        accuracy = overlap_modularity.overlapping_modularity(e_set, e_count, prediction)
        print("OM:", accuracy)
        accuracies.append(accuracy)


    result_table = PrettyTable()
    result_table.field_names = ["PERCENTAGE", "ONMI", "OM"]
    c = 0
    for i in range(0, len(accuracies), 2):
        result_table.add_row([
        f"{percentages[c]:.2f}",
        f"{accuracies[i]:.2f}",
        f"{accuracies[i+1]:.2f}"
        ])
        c += 1
    print(result_table)