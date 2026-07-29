import networkx as nx


def load_nodes(node_file):
    """
    Load all nodes from CS-Aarhus_nodes.txt
    """
    nodes = []

    with open(node_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            parts = line.split()

            if parts[0].isdigit():
                nodes.append(int(parts[0]))

    return nodes

def load_multiplex_dataset(edge_file, node_file):
    """
    Load multiplex network and create
    separate graphs for each layer.
    """

    all_nodes = load_nodes(node_file)

    layers = {}

    # Create graphs
    with open(edge_file, "r") as f:
        for line in f:

            if line.startswith("#") or line.strip() == "":
                continue

            layer_id, node1, node2, weight = line.split()

            layer_id = int(layer_id)
            node1 = int(node1)
            node2 = int(node2)

            if layer_id not in layers:
                layers[layer_id] = nx.Graph()

            layers[layer_id].add_edge(
                node1,
                node2,
                weight=int(weight)
            )

    # Add all nodes to every layer
    for layer_id, graph in layers.items():
        graph.add_nodes_from(all_nodes)

    return layers


if __name__ == "__main__":

    edge_path = "data/raw/CS-Aarhus_multiplex.edges"
    node_path = "data/raw/CS-Aarhus_nodes.txt"

    graphs = load_multiplex_dataset(edge_path, node_path)

    print("Number of layers:", len(graphs))

    for layer_id, graph in graphs.items():
        print(
            f"Layer {layer_id}: "
            f"Nodes={graph.number_of_nodes()}, "
            f"Edges={graph.number_of_edges()}"
        )