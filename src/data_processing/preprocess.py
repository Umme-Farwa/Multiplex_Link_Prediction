import pickle
from load_dataset import load_multiplex_dataset


def load_layer_names(layer_file):
    """
    Load layer IDs and names from CS-Aarhus_layers.txt
    """

    layers = {}

    with open(layer_file, "r") as f:
        for line in f:

            if line.strip() == "":
                continue

            parts = line.split()

            # Skip header line
            if not parts[0].isdigit():
                continue

            layer_id = int(parts[0])

            # Remaining parts are the layer name
            layer_name = " ".join(parts[1:])

            layers[layer_id] = layer_name

    return layers


if __name__ == "__main__":

    edge_path = "data/raw/CS-Aarhus_multiplex.edges"
    node_path = "data/raw/CS-Aarhus_nodes.txt"
    layer_path = "data/raw/CS-Aarhus_layers.txt"


    # Load graphs
    graphs = load_multiplex_dataset(
        edge_path,
        node_path
    )


    # Load layer names
    layer_names = load_layer_names(layer_path)


    multiplex_graph = {}


    for layer_id, graph in graphs.items():

        layer_name = layer_names[layer_id]

        multiplex_graph[layer_name] = graph

        print(
            f"{layer_id}: {layer_name} | "
            f"Nodes={graph.number_of_nodes()}, "
            f"Edges={graph.number_of_edges()}"
        )


    # Save processed multiplex network
    with open(
        "data/processed/multiplex_graphs.pkl",
        "wb"
    ) as f:
        pickle.dump(multiplex_graph, f)


    print("\nProcessed multiplex dataset saved!")