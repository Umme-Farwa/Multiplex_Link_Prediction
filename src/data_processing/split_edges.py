import pickle
import random
import networkx as nx



def split_edges(graph, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Split weighted edges into train, validation and test sets.
    """

    random.seed(seed)

    edges = list(
        graph.edges(data=True)
    )

    random.shuffle(edges)


    total_edges = len(edges)

    test_size = int(total_edges * test_ratio)

    val_size = int(total_edges * val_ratio)


    test_edges = edges[:test_size]


    val_edges = edges[
        test_size:test_size + val_size
    ]


    train_edges = edges[
        test_size + val_size:
    ]


    return train_edges, val_edges, test_edges





def create_train_graph(nodes, train_edges):
    """
    Create graph only from training edges.
    """

    G = nx.Graph()

    G.add_nodes_from(nodes)


    for u, v, data in train_edges:

        G.add_edge(
            u,
            v,
            weight=data["weight"]
        )


    return G





def generate_negative_edges(graph, number_of_edges, seed=42):
    """
    Generate non-existing edges.
    """

    random.seed(seed)


    nodes = list(
        graph.nodes()
    )


    negative_edges = set()


    while len(negative_edges) < number_of_edges:


        u = random.choice(nodes)

        v = random.choice(nodes)


        if u == v:
            continue


        if graph.has_edge(u, v):
            continue


        negative_edges.add(
            tuple(sorted((u, v)))
        )


    return list(negative_edges)






if __name__ == "__main__":


    with open(
        "data/processed/multiplex_graphs.pkl",
        "rb"
    ) as f:

        graphs = pickle.load(f)




    train_data = {}
    val_data = {}
    test_data = {}

    val_negative_data = {}
    test_negative_data = {}




    for layer_name, graph in graphs.items():


        print("\nLayer:", layer_name)



        train_edges, val_edges, test_edges = split_edges(
            graph,
            test_ratio=0.2,
            val_ratio=0.1,
            seed=42
        )



        # Training graph (no validation/test edges)
        train_graph = create_train_graph(
            graph.nodes(),
            train_edges
        )



        val_negative_edges = generate_negative_edges(
            train_graph,
            len(val_edges),
            seed=42
        )


        test_negative_edges = generate_negative_edges(
            train_graph,
            len(test_edges),
            seed=43
        )



        train_data[layer_name] = train_edges

        val_data[layer_name] = val_edges

        test_data[layer_name] = test_edges


        val_negative_data[layer_name] = val_negative_edges

        test_negative_data[layer_name] = test_negative_edges




        print(
            "Train edges:",
            len(train_edges)
        )

        print(
            "Validation edges:",
            len(val_edges)
        )

        print(
            "Test edges:",
            len(test_edges)
        )

        print(
            "Validation negative:",
            len(val_negative_edges)
        )

        print(
            "Test negative:",
            len(test_negative_edges)
        )






    with open(
        "data/processed/train_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(train_data, f)



    with open(
        "data/processed/val_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(val_data, f)



    with open(
        "data/processed/test_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(test_data, f)



    with open(
        "data/processed/val_negative_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(val_negative_data, f)



    with open(
        "data/processed/test_negative_edges.pkl",
        "wb"
    ) as f:
        pickle.dump(test_negative_data, f)

# Generate train negative edges

train_negative_data = {}


for layer_name, graph in graphs.items():

    print("\nGenerating train negatives for:", layer_name)

    train_negative_edges = generate_negative_edges(
        graph,
        len(train_data[layer_name]),
        seed=44
    )

    train_negative_data[layer_name] = train_negative_edges


    print(
        "Train negative edges:",
        len(train_negative_edges)
    )



# Save train negative edges

with open(
    "data/processed/train_negative_edges.pkl",
    "wb"
) as f:

    pickle.dump(
        train_negative_data,
        f
    )


print("\nTrain negative edges saved successfully!")


print("\nEdge splitting completed successfully!")