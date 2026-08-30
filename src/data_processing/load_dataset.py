import pandas as pd
import networkx as nx


def load_multiplex_dataset(edge_path):
    """
    Load arXiv multiplex network and create
    layer-wise NetworkX graphs.
    """

    # Read raw edge file
    edges = pd.read_csv(
        edge_path,
        sep=r"\s+",
        header=None,
        names=[
            "layer",
            "source",
            "target",
            "weight"
        ]
    )


    multiplex_graphs = {}


    # Create graph for each layer
    for layer_id in edges["layer"].unique():

        layer_edges = edges[
            edges["layer"] == layer_id
        ]


        G = nx.Graph()


        for _, row in layer_edges.iterrows():

            G.add_edge(
                int(row["source"]),
                int(row["target"]),
                weight=float(row["weight"])
            )


        multiplex_graphs[layer_id] = G


    return multiplex_graphs
