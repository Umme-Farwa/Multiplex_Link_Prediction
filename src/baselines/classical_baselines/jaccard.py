import pickle

from src.evaluation.metrics import evaluate_all
from src.utils.save_results import (
    save_metrics,
    save_predictions
)



def jaccard_score(graph, node1, node2):
    """
    Calculate Jaccard similarity score.
    """

    neighbors_1 = set(graph.neighbors(node1))
    neighbors_2 = set(graph.neighbors(node2))


    union = neighbors_1.union(neighbors_2)

    if len(union) == 0:
        return 0.0


    intersection = neighbors_1.intersection(
        neighbors_2
    )


    return len(intersection) / len(union)



def create_training_graph(original_graph, train_edges):
    """
    Create graph using only training edges.
    """

    import networkx as nx

    train_graph = nx.Graph()

    train_graph.add_nodes_from(
        original_graph.nodes()
    )

    train_graph.add_edges_from(
        train_edges
    )

    return train_graph



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

            score = jaccard_score(
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

            score = jaccard_score(
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
        "classical_metrics/jaccard_metrics.csv"
    )


    save_predictions(
        all_predictions,
        "classical_predictions/jaccard_predictions.csv"
    )


    print("\nResults saved successfully!")