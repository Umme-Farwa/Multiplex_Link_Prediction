import sys
import os
import pickle
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



def jaccard_score(graph, node1, node2):
    """
    Calculate Jaccard Coefficient score.
    """

    neighbors_1 = set(graph.neighbors(node1))
    neighbors_2 = set(graph.neighbors(node2))


    union = neighbors_1.union(neighbors_2)


    if len(union) == 0:
        return 0


    intersection = neighbors_1.intersection(neighbors_2)


    return len(intersection) / len(union)




def create_training_graph(original_graph, train_edges):

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




def score_edge_set(train_graph, positive_edges, negative_edges):
    """
    Score a set of positive and negative edges with the Jaccard
    coefficient. Returns (labels, scores): positives first, then
    negatives. Used for both validation (threshold tuning) and test.
    """

    labels = []
    scores = []


    for edge in positive_edges:

        u = edge[0]
        v = edge[1]

        scores.append(
            jaccard_score(train_graph, u, v)
        )
        labels.append(1)


    for edge in negative_edges:

        u = edge[0]
        v = edge[1]

        scores.append(
            jaccard_score(train_graph, u, v)
        )
        labels.append(0)


    return labels, scores




if __name__ == "__main__":


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



    # Validation edges (used ONLY to tune the F1 threshold)

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



    with open(
        "data/processed/test_edges.pkl",
        "rb"
    ) as f:
        test_edges = pickle.load(f)



    with open(
        "data/processed/test_negative_edges.pkl",
        "rb"
    ) as f:
        test_negative_edges = pickle.load(f)



    all_metrics = []
    all_predictions = []



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
        # Evaluation (F1 uses validation-tuned threshold)
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

        all_metrics.append(results)

        all_predictions.extend(predictions)



        print("\nEvaluation Results:")


        for metric, value in results.items():

            if metric != "Layer":

                print(
                    f"{metric}: {value:.4f}"
                )




    save_metrics(
        all_metrics,
        "classical/jaccard_metrics.csv"
    )


    save_predictions(
        all_predictions,
        "classical/jaccard_predictions.csv"
    )


    print("\nJaccard baseline completed successfully!")