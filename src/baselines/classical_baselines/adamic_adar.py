import sys
import os
import pickle
import math
import networkx as nx


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../.."
        )
    )
)


from src.evaluation.metrics import evaluate_all, find_best_threshold
from src.utils.save_results import (
    save_metrics,
    save_predictions
)



def create_training_graph(original_graph, train_edges):
    """
    Create graph using only training edges.
    Avoids information leakage.
    """

    train_graph = nx.Graph()

    train_graph.add_nodes_from(
        original_graph.nodes()
    )


    for u, v, weight in train_edges:

        train_graph.add_edge(
            u,
            v,
            weight=weight
        )


    return train_graph




def adamic_adar_score(graph, node1, node2):
    """
    Calculate Adamic-Adar score.

    AA(u,v) = sum(1/log(degree(z)))
    for common neighbors z
    """

    score = 0.0


    neighbors_1 = set(
        graph.neighbors(node1)
    )

    neighbors_2 = set(
        graph.neighbors(node2)
    )


    common_neighbors = (
        neighbors_1.intersection(
            neighbors_2
        )
    )


    for neighbor in common_neighbors:

        degree = graph.degree(
            neighbor
        )


        if degree > 1:

            score += (
                1 / math.log(degree)
            )


    return score




def score_edge_set(train_graph, positive_edges, negative_edges):
    """
    Score a set of positive and negative edges with Adamic-Adar.

    Returns (labels, scores) aligned as: positives first, then
    negatives. Works for both validation (threshold tuning) and
    test (final evaluation), so the scoring logic is not duplicated.

    Note: positive edges are (u, v, weight) triples while negative
    edges are (u, v) pairs -- indexing edge[0]/edge[1] handles both.
    """

    labels = []
    scores = []


    for edge in positive_edges:

        u = edge[0]
        v = edge[1]

        scores.append(
            adamic_adar_score(train_graph, u, v)
        )
        labels.append(1)


    for edge in negative_edges:

        u = edge[0]
        v = edge[1]

        scores.append(
            adamic_adar_score(train_graph, u, v)
        )
        labels.append(0)


    return labels, scores




if __name__ == "__main__":


    # Load multiplex graphs

    with open(
        "data/processed/multiplex_graphs.pkl",
        "rb"
    ) as f:

        graphs = pickle.load(f)



    # Load train edges

    with open(
        "data/processed/train_edges.pkl",
        "rb"
    ) as f:

        train_edges = pickle.load(f)



    # Load validation edges (used ONLY to tune the F1 threshold)

    with open(
        "data/processed/val_edges.pkl",
        "rb"
    ) as f:

        val_edges = pickle.load(f)



    with open(
        "data/processed/val_negative_edges.pkl",
        "rb"
    ) as f:

        val_negative_edges = pickle.load(f)



    # Load positive test edges

    with open(
        "data/processed/test_edges.pkl",
        "rb"
    ) as f:

        test_edges = pickle.load(f)



    # Load negative test edges

    with open(
        "data/processed/test_negative_edges.pkl",
        "rb"
    ) as f:

        test_negative_edges = pickle.load(f)




    all_metrics = []

    all_predictions = []




    # Run Adamic-Adar for every layer

    for layer_name, graph in graphs.items():


        print("\n====================")
        print("Layer:", layer_name)
        print("====================")



        train_graph = create_training_graph(
            graph,
            train_edges[layer_name]
        )



        # ------------------------------------------
        # VALIDATION: tune the F1 threshold
        # (leakage-free -- test data is never used here)
        # ------------------------------------------

        val_labels, val_scores = score_edge_set(
            train_graph,
            val_edges[layer_name],
            val_negative_edges[layer_name]
        )


        best_threshold, best_val_f1 = find_best_threshold(
            val_labels,
            val_scores
        )


        print(
            f"\nBest threshold (from val): {best_threshold:.4f} "
            f"(val F1 = {best_val_f1:.4f})"
        )



        # ------------------------------------------
        # TEST: score positives + negatives
        # ------------------------------------------

        test_labels, test_scores = score_edge_set(
            train_graph,
            test_edges[layer_name],
            test_negative_edges[layer_name]
        )



        # Build predictions (positives first, then negatives),
        # aligned with the order in test_scores

        predictions = []

        index = 0


        for u, v, weight in test_edges[layer_name]:

            predictions.append(
                (u, v, test_scores[index], 1, layer_name)
            )

            index += 1


        for u, v in test_negative_edges[layer_name]:

            predictions.append(
                (u, v, test_scores[index], 0, layer_name)
            )

            index += 1



        # Top predicted links

        print("\nTop predicted links:")


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



        # ------------------------------------------
        # Evaluation
        # (F1 uses the validation-tuned threshold so it is
        #  comparable to the GNN / proposed models)
        # ------------------------------------------

        results = evaluate_all(
            test_labels,
            test_scores,
            threshold=best_threshold
        )



        results = {
            "ROC-AUC": results["ROC-AUC"],
            "PR-AUC": results["PR-AUC"],
            "F1-score": results["F1-score"],
            "Threshold_Used": best_threshold,
            "Layer": layer_name
        }



        all_metrics.append(
            results
        )


        all_predictions.extend(
            predictions
        )



        print("\nEvaluation Results:")


        for metric, value in results.items():

            if metric != "Layer":

                print(
                    f"{metric}: {value:.4f}"
                )




    # Save metrics

    save_metrics(
        all_metrics,
        "classical/adamic_adar_metrics.csv"
    )



    # Save predictions

    save_predictions(
        all_predictions,
        "classical/adamic_adar_predictions.csv"
    )



    print(
        "\nAdamic-Adar baseline completed successfully!"
    )