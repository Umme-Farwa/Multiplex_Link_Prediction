import pickle
import networkx as nx

from src.evaluation.metrics import evaluate_all
from src.utils.save_results import (
    save_metrics,
    save_predictions
)



def create_training_graph(original_graph, train_edges):
    """
    Create graph using only training edges.
    """

    train_graph = nx.Graph()

    train_graph.add_nodes_from(
        original_graph.nodes()
    )

    train_graph.add_edges_from(
        train_edges
    )

    return train_graph



def adamic_adar_score(graph, node1, node2):
    """
    Calculate Adamic-Adar score.
    """

    score = 0.0

    common_neighbors = set(
        graph.neighbors(node1)
    ).intersection(
        set(graph.neighbors(node2))
    )


    for neighbor in common_neighbors:

        degree = graph.degree(neighbor)

        if degree > 1:

            score += 1 / __import__("math").log(degree)


    return score



if __name__ == "__main__":


    # Load processed data

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



    for layer_name, graph in graphs.items():

        print("\n====================")
        print("Layer:", layer_name)
        print("====================")


        train_graph = create_training_graph(
            graph,
            train_edges[layer_name]
        )


        labels = []
        scores = []
        predictions = []



        # Positive links

        for u, v in test_edges[layer_name]:

            score = adamic_adar_score(
                train_graph,
                u,
                v
            )

            scores.append(score)
            labels.append(1)

            predictions.append(
                (u, v, score, 1, layer_name)
            )



        # Negative links

        for u, v in negative_edges[layer_name]:

            score = adamic_adar_score(
                train_graph,
                u,
                v
            )

            scores.append(score)
            labels.append(0)

            predictions.append(
                (u, v, score, 0, layer_name)
            )



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



        # Evaluation

        results = evaluate_all(
            labels,
            scores
        )


        results["Layer"] = layer_name

        all_metrics.append(results)

        all_predictions.extend(predictions)



        print("\nEvaluation Results:")

        for metric, value in results.items():

            if metric != "Layer":

                print(
                    f"{metric}: {value:.4f}"
                )



    # Save results

    save_metrics(
        all_metrics,
        "classical_metrics/adamic_adar_metrics.csv"
    )


    save_predictions(
        all_predictions,
        "classical_predictions/adamic_adar_predictions.csv"
    )


    print("\nResults saved successfully!")