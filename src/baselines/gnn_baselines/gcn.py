import pickle
import random

import torch
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.evaluation.metrics import evaluate_all
from src.utils.save_results import (
    save_metrics,
    save_predictions
)



# ==========================================
# Negative edges generator (training)
# ==========================================

def generate_negative_edges(graph, number_of_edges, seed=42):

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




# ==========================================
# Training graph
# ==========================================

def create_training_graph(
        original_graph,
        train_edges
):

    train_graph = nx.Graph()


    train_graph.add_nodes_from(
        original_graph.nodes()
    )


    train_graph.add_edges_from(
        train_edges
    )


    return train_graph




# ==========================================
# NetworkX -> PyG
# ==========================================

def convert_to_pyg(graph):


    graph = nx.convert_node_labels_to_integers(
        graph,
        ordering="sorted"
    )


    edges = list(graph.edges())


    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()



    num_nodes = graph.number_of_nodes()


    # Identity node features
    
    degrees= dict(graph.degree())
    x = torch.tensor([[degrees[i]] for i in range(num_nodes)],
    dtype=torch.float
    )
    

    data = Data(
        x=x,
        edge_index=edge_index
    )


    return data




# ==========================================
# GCN Model
# ==========================================

class GCN(torch.nn.Module):


    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout=0.5
    ):

        super().__init__()


        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )


        self.conv2 = GCNConv(
            hidden_dim,
            output_dim
        )


        self.dropout = dropout



    def forward(
            self,
            x,
            edge_index
    ):


        x = self.conv1(
            x,
            edge_index
        )


        x = F.relu(x)


        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )


        return x




# ==========================================
# Link decoder
# ==========================================

def decode(
        embeddings,
        edges
):

    scores = []


    for u, v in edges:


        score = torch.sum(
            embeddings[u] *
            embeddings[v]
        )


        scores.append(score)



    return torch.stack(scores)




# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Using device:",
        device
    )



    # Load data

    with open(
        "data/processed/multiplex_graphs.pkl",
        "rb"
    ) as f:

        graphs = pickle.load(f)



    with open(
        "data/processed/train_edges.pkl",
        "rb"
    ) as f:

        train_edges = pickle.load(f)



    with open(
        "data/processed/test_edges.pkl",
        "rb"
    ) as f:

        test_edges = pickle.load(f)



    with open(
        "data/processed/negative_edges.pkl",
        "rb"
    ) as f:

        negative_edges = pickle.load(f)



    all_metrics = []

    all_predictions = []




    # ==================================
    # Each multiplex layer
    # ==================================

    for layer_name, graph in graphs.items():


        print("\n====================")
        print("Layer:", layer_name)
        print("====================")



        train_graph = create_training_graph(
            graph,
            train_edges[layer_name]
        )



        data = convert_to_pyg(
            train_graph
        )


        data = data.to(device)



        node_map = {

            node: idx

            for idx, node in enumerate(
                sorted(train_graph.nodes())
            )

        }




        # ------------------------------
        # Training samples
        # ------------------------------


        train_positive = [

            (
                node_map[u],
                node_map[v]
            )

            for u, v in train_edges[layer_name]

        ]



        train_negative_raw = generate_negative_edges(
            train_graph,
            len(train_positive)
        )



        train_negative = [

            (
                node_map[u],
                node_map[v]
            )

            for u, v in train_negative_raw

        ]



        train_samples = (
            train_positive +
            train_negative
        )



        train_labels = (

            [1] * len(train_positive)

            +

            [0] * len(train_negative)

        )



        train_labels = torch.tensor(
            train_labels,
            dtype=torch.float,
            device=device
        )




        # ------------------------------
        # Model
        # ------------------------------


        model = GCN(
            data.x.shape[1],
            16,
            8,
            dropout=0.5
        ).to(device)



        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,
            weight_decay=5e-4
        )


        criterion = torch.nn.BCEWithLogitsLoss()




        # ------------------------------
        # Training + Early stopping
        # ------------------------------


        best_loss = float("inf")
        patience = 20
        counter = 0


        model.train()


        for epoch in range(50):


            optimizer.zero_grad()


            embeddings = model(
                data.x,
                data.edge_index
            )


            train_scores = decode(
                embeddings,
                train_samples
            )


            loss = criterion(
                train_scores,
                train_labels
            )


            loss.backward()


            optimizer.step()



            if loss.item() < best_loss:

                best_loss = loss.item()
                counter = 0

            else:

                counter += 1



            if counter >= patience:

                print(
                    "Early stopping at epoch:",
                    epoch
                )

                break




        # ------------------------------
        # Training evaluation
        # ------------------------------


        model.eval()


        with torch.no_grad():

            embeddings = model(
                data.x,
                data.edge_index
            )



        train_scores = decode(
            embeddings,
            train_samples
        )


        train_scores = torch.sigmoid(
            train_scores
        ).cpu().numpy()



        train_results = evaluate_all(
            train_labels.cpu().numpy(),
            train_scores
        )



        print("\nTraining Results:")


        for metric, value in train_results.items():

            print(
                f"{metric}: {value:.4f}"
            )




        # ------------------------------
        # Testing
        # ------------------------------


        test_samples_raw = (

            test_edges[layer_name]

            +

            negative_edges[layer_name]

        )



        test_samples = [

            (
                node_map[u],
                node_map[v]
            )

            for u, v in test_samples_raw

        ]



        test_scores = decode(
            embeddings,
            test_samples
        )


        test_scores = torch.sigmoid(
            test_scores
        ).cpu().numpy()



        test_labels = (

            [1] * len(test_edges[layer_name])

            +

            [0] * len(negative_edges[layer_name])

        )




        # ------------------------------
        # Predictions
        # ------------------------------


        print("\nTop predicted links:")


        predictions = []


        for (u, v), score, label in zip(
                test_samples_raw,
                test_scores,
                test_labels
        ):


            predictions.append(
                (
                    u,
                    v,
                    float(score),
                    label,
                    layer_name
                )
            )



        top_predictions = sorted(
            predictions,
            key=lambda x: x[2],
            reverse=True
        )[:10]



        for u, v, score, label, layer in top_predictions:


            print(
                f"Nodes ({u},{v}) "
                f"Score={score:.4f} "
                f"Label={label}"
            )



        all_predictions.extend(
            predictions
        )




        # ------------------------------
        # Test metrics
        # ------------------------------


        results = evaluate_all(
            test_labels,
            test_scores
        )


        results["Layer"] = layer_name


        all_metrics.append(
            results
        )



        print("\nTesting Results:")


        for metric, value in results.items():

            if metric != "Layer":

                print(
                    f"{metric}: {value:.4f}"
                )




    # ==================================
    # Save
    # ==================================


    save_metrics(
        all_metrics,
        "gnn_metrics/gcn_metrics.csv"
    )


    save_predictions(
        all_predictions,
        "gnn_predictions/gcn_predictions.csv"
    )


    print(
        "\nGCN baseline completed!"
    )

    print(
        "Metrics and predictions saved!"
    )