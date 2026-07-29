import pickle
import random
import networkx as nx


def split_edges(graph, test_ratio=0.2, seed=42):
    """
    Split edges into train and test sets.
    """

    random.seed(seed)

    edges = list(graph.edges())

    random.shuffle(edges)

    test_size = int(len(edges) * test_ratio)

    test_edges = edges[:test_size]
    train_edges = edges[test_size:]

    return train_edges, test_edges



def generate_negative_edges(graph, number_of_edges, seed=42):
    """
    Generate non-existing edges.
    """

    random.seed(seed)

    nodes = list(graph.nodes())

    negative_edges = []

    while len(negative_edges) < number_of_edges:

        u = random.choice(nodes)
        v = random.choice(nodes)

        if u == v:
            continue

        if graph.has_edge(u, v):
            continue

        if (u, v) in negative_edges or (v, u) in negative_edges:
            continue

        negative_edges.append((u, v))

    return negative_edges



if __name__ == "__main__":

    # Load processed multiplex graphs

    with open(
        "data/processed/multiplex_graphs.pkl",
        "rb"
    ) as f:
        graphs = pickle.load(f)


    train_data = {}
    test_data = {}
    negative_data = {}


    for layer_name, graph in graphs.items():

        train_edges, test_edges = split_edges(graph)

        negative_edges = generate_negative_edges(
            graph,
            len(test_edges)
        )


        train_data[layer_name] = train_edges
        test_data[layer_name] = test_edges
        negative_data[layer_name] = negative_edges


        print(
            f"{layer_name}: "
            f"Train={len(train_edges)}, "
            f"Test={len(test_edges)}, "
            f"Negative={len(negative_edges)}"
        )


    # Save splits

    with open(
        "data/processed/train_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(train_data, f)


    with open(
        "data/processed/test_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(test_data, f)


    with open(
        "data/processed/negative_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(negative_data, f)


    print("\nEdge splitting completed!")