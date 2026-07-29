import os
import csv
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)


# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)



# ==========================
# PATHS
# ==========================

GRAPH_PATH = "data/processed/multiplex_graphs.pkl"
TRAIN_PATH = "data/processed/train_edges.pkl"
TEST_PATH = "data/processed/test_edges.pkl"
NEG_PATH = "data/processed/negative_edges.pkl"


METRIC_PATH = "experiments/baseline_results/metrics/gnn_metrics/gat_metrics.csv"

PRED_PATH = "experiments/baseline_results/predictions/gnn_predictions/gat_predictions.csv"



# ==========================
# LOAD DATA
# ==========================

with open(GRAPH_PATH, "rb") as f:
    graphs = pickle.load(f)


with open(TRAIN_PATH, "rb") as f:
    train_edges = pickle.load(f)


with open(TEST_PATH, "rb") as f:
    test_edges = pickle.load(f)


with open(NEG_PATH, "rb") as f:
    negative_edges = pickle.load(f)



# ==========================
# GAT MODEL
# ==========================

class GAT(torch.nn.Module):

    def __init__(self):

        super().__init__()


        self.conv1 = GATConv(
            in_channels=1,
            out_channels=16,
            heads=2,
            dropout=0.4
        )


        self.conv2 = GATConv(
            in_channels=32,
            out_channels=8,
            heads=1,
            dropout=0.4
        )



    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )


        x = F.elu(x)


        x = F.dropout(
            x,
            p=0.4,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index
        )


        return x



# ==========================
# LINK PREDICTION DECODER
# ==========================

def decode(
        z,
        edges
):

    src = z[edges[0]]

    dst = z[edges[1]]


    score = (
        src * dst
    ).sum(dim=1)


    return torch.sigmoid(score)



# ==========================
# EVALUATION
# ==========================

def evaluate(
        model,
        data,
        positive_edges,
        negative_edges
):

    model.eval()


    with torch.no_grad():

        z = model(
            data.x,
            data.edge_index
        )


        pos_scores = decode(
            z,
            positive_edges
        )


        neg_scores = decode(
            z,
            negative_edges
        )



    scores = torch.cat(
        [
            pos_scores,
            neg_scores
        ]
    ).cpu()



    labels = torch.cat(
        [
            torch.ones(
                len(pos_scores)
            ),

            torch.zeros(
                len(neg_scores)
            )
        ]
    )



    auc = roc_auc_score(
        labels,
        scores
    )


    pr = average_precision_score(
        labels,
        scores
    )


    prediction = (
        scores >= 0.5
    ).int()


    f1 = f1_score(
        labels,
        prediction
    )


    # Hits@10

    k = min(
        10,
        len(scores)
    )


    top_indices = torch.topk(
        scores,
        k
    ).indices


    hits10 = (
        labels[top_indices].sum().item()
        /
        k
    )


    # MRR

    ranking = torch.argsort(
        scores,
        descending=True
    )


    rank = None


    for i, idx in enumerate(ranking):

        if labels[idx] == 1:

            rank = i + 1
            break



    if rank is None:

        mrr = 0

    else:

        mrr = 1 / rank



    metrics = {

        "ROC-AUC": auc,

        "PR-AUC": pr,

        "F1-score": f1,

        "Hits@10": hits10,

        "MRR": mrr
    }


    return (
        metrics,
        scores,
        labels
    )

# ==========================
# CSV FILES
# ==========================

metric_file = open(
    METRIC_PATH,
    "w",
    newline=""
)

metric_writer = csv.writer(
    metric_file
)

metric_writer.writerow(
    [
        "Layer",
        "ROC-AUC",
        "PR-AUC",
        "F1-score",
        "Hits@10",
        "MRR"
    ]
)



pred_file = open(
    PRED_PATH,
    "w",
    newline=""
)

pred_writer = csv.writer(
    pred_file
)

pred_writer.writerow(
    [
        "node1",
        "node2",
        "score",
        "label",
        "layer"
    ]
)



# ==========================
# EDGE CONVERTER
# ==========================

def convert_edges(edges):

    return torch.tensor(
        [
            [
                u - 1
                for u, v in edges
            ],

            [
                v - 1
                for u, v in edges
            ]
        ],
        dtype=torch.long
    ).to(device)




# ==========================
# TRAIN EACH MULTIPLEX LAYER
# ==========================

for layer, graph in graphs.items():

    print("\n====================")
    print("Layer:", layer)
    print("====================")



    # Graph edges

    graph_edges = list(
        graph.edges()
    )


    edge_index = torch.tensor(
        [
            [
                u - 1
                for u, v in graph_edges
            ],

            [
                v - 1
                for u, v in graph_edges
            ]
        ],
        dtype=torch.long
    )



    print(
        "Max edge index:",
        edge_index.max().item()
    )



    # Node features (degree)

    num_nodes = max(
        graph.nodes()
    )


    degree = torch.zeros(
        num_nodes
    )


    for u, v in graph_edges:

        degree[u-1] += 1
        degree[v-1] += 1



    x = degree.view(
        -1,
        1
    )



    data = Data(
        x=x,
        edge_index=edge_index
    ).to(device)



    train_e = convert_edges(
        train_edges[layer]
    )


    test_e = convert_edges(
        test_edges[layer]
    )


    neg_e = convert_edges(
        negative_edges[layer]
    )



    # ======================
    # MODEL
    # ======================

    model = GAT().to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )



    best_auc = 0

    patience = 10

    counter = 0



    # ======================
    # TRAINING
    # ======================

    for epoch in range(100):

        model.train()


        optimizer.zero_grad()


        z = model(
            data.x,
            data.edge_index
        )


        pos_score = decode(
            z,
            train_e
        )


        neg_score = decode(
            z,
            neg_e
        )


        loss = (

            -torch.log(
                pos_score + 1e-15
            ).mean()

            -

            torch.log(
                1 - neg_score + 1e-15
            ).mean()

        )


        loss.backward()

        optimizer.step()



        val_metrics, _, _ = evaluate(
            model,
            data,
            test_e,
            neg_e
        )



        current_auc = val_metrics["ROC-AUC"]



        if current_auc > best_auc:

            best_auc = current_auc
            counter = 0

        else:

            counter += 1



        if counter >= patience:

            print(
                "Early stopping at epoch:",
                epoch
            )

            break




    # ======================
    # FINAL TEST
    # ======================

    metrics, scores, labels = evaluate(
        model,
        data,
        test_e,
        neg_e
    )


    print(
        "\nTesting Results:"
    )


    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )



    # ======================
    # SAVE METRICS
    # ======================

    metric_writer.writerow(
        [
            layer,
            round(metrics["ROC-AUC"],4),
            round(metrics["PR-AUC"],4),
            round(metrics["F1-score"],4),
            round(metrics["Hits@10"],4),
            round(metrics["MRR"],4)
        ]
    )



    # ======================
    # PREDICTIONS
    # ======================

    all_edges = torch.cat(
        [
            test_e,
            neg_e
        ],
        dim=1
    ).cpu()



    print(
        "\nTop predicted links:"
    )


    top_k = min(
        10,
        len(scores)
    )


    top_indices = torch.topk(
        scores,
        top_k
    ).indices



    for idx in top_indices:

        node1 = int(
            all_edges[0,idx]
        ) + 1


        node2 = int(
            all_edges[1,idx]
        ) + 1


        print(
            f"Nodes ({node1},{node2}) "
            f"Score={scores[idx]:.4f} "
            f"Label={int(labels[idx])}"
        )



    for i in range(
        all_edges.shape[1]
    ):

        pred_writer.writerow(
            [
                int(all_edges[0,i]) + 1,
                int(all_edges[1,i]) + 1,
                round(float(scores[i]),6),
                int(labels[i]),
                layer
            ]
        )



metric_file.close()

pred_file.close()



print("\nGAT baseline completed!")
print("Metrics and predictions saved!")