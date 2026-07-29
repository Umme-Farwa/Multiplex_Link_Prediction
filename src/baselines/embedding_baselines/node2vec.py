import pickle
import numpy as np

from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.metrics import evaluate_all
from src.utils.save_results import (
    save_metrics,
    save_predictions
)


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



def edge_score(embeddings, node1, node2):
    """
    Calculate link score using cosine similarity
    between node embeddings.
    """

    vec1 = embeddings[str(node1)].reshape(1, -1)
    vec2 = embeddings[str(node2)].reshape(1, -1)

    score = cosine_similarity(
        vec1,
        vec2
    )[0][0]

    return score



if __name__ == "__main__":


    # -----------------------------
    # Load processed data
    # -----------------------------

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



    # -----------------------------
    # Run Node2Vec per layer
    # -----------------------------

    for layer_name, graph in graphs.items():

        print("\n====================")
        print("Layer:", layer_name)
        print("====================")


        # Create training graph

        train_graph = create_training_graph(
            graph,
            train_edges[layer_name]
        )


        # -----------------------------
        # Generate embeddings
        # -----------------------------

        node2vec = Node2Vec(
            train_graph,
            dimensions=64,
            walk_length=10,
            num_walks=100,
            workers=1,
            seed=42
        )


        model = node2vec.fit(
            window=5,
            min_count=1,
            batch_words=4
        )


        embeddings = {
            str(node): model.wv[str(node)]
            for node in train_graph.nodes()
        }



        labels = []
        scores = []
        predictions = []



        # -----------------------------
        # Positive test links
        # -----------------------------

        for u, v in test_edges[layer_name]:

            score = edge_score(
                embeddings,
                u,
                v
            )

            scores.append(score)
            labels.append(1)

            predictions.append(
                (u, v, score, 1, layer_name)
            )



        # -----------------------------
        # Negative links
        # -----------------------------

        for u, v in negative_edges[layer_name]:

            score = edge_score(
                embeddings,
                u,
                v
            )

            scores.append(score)
            labels.append(0)

            predictions.append(
                (u, v, score, 0, layer_name)
            )



        # -----------------------------
        # Top predictions
        # -----------------------------

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



        # -----------------------------
        # Evaluation
        # -----------------------------

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



    # -----------------------------
    # Save results
    # -----------------------------

    save_metrics(
        all_metrics,
        "embedding_metrics/node2vec_metrics.csv"
    )


    save_predictions(
        all_predictions,
        "embedding_predictions/node2vec_predictions.csv"
    )


    print("\nNode2Vec results saved successfully!")