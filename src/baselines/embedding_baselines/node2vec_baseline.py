import sys
import os
import pickle
import numpy as np
import networkx as nx


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../.."
        )
    )
)


from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression


from src.evaluation.metrics import evaluate_all, find_best_threshold
from src.utils.save_results import (
    save_metrics,
    save_predictions
)



def create_training_graph(original_graph, train_edges):
    """
    Create unweighted training graph for Node2Vec.
    Only training edges are used.
    """

    train_graph = nx.Graph()

    train_graph.add_nodes_from(
        original_graph.nodes()
    )


    for edge in train_edges:

        u = edge[0]
        v = edge[1]


        train_graph.add_edge(
            u,
            v,
            weight=1.0
        )


    return train_graph




def create_edge_embedding(vec1, vec2):
    """
    Hadamard product edge embedding.
    """

    return vec1 * vec2




def build_edge_features(embedding_model, positive_edges, negative_edges):
    """
    Build (X, y) edge-feature matrix for a set of positive and
    negative edges, using Hadamard-product node embeddings.
    Order: positives first, then negatives.
    """

    X = []
    y = []

    for edge in positive_edges:

        u = edge[0]
        v = edge[1]

        edge_vector = create_edge_embedding(
            embedding_model.wv[str(u)],
            embedding_model.wv[str(v)]
        )

        X.append(edge_vector)
        y.append(1)

    for edge in negative_edges:

        u = edge[0]
        v = edge[1]

        edge_vector = create_edge_embedding(
            embedding_model.wv[str(u)],
            embedding_model.wv[str(v)]
        )

        X.append(edge_vector)
        y.append(0)

    return np.array(X), np.array(y)




if __name__ == "__main__":

    # --- multi-seed support (backward compatible) ---
    # THESIS_SEED    : which random seed to use (default 42 = original run)
    # THESIS_RUN_TAG : if set, results go to experiments/multiseed/ instead
    #                  of overwriting the canonical baseline results
    # (Node2Vec has no training epoch loop, so no THESIS_MAX_EPOCHS here.)
    import random
    SEED = int(os.environ.get("THESIS_SEED", "42"))
    RUN_TAG = os.environ.get("THESIS_RUN_TAG", "")
    random.seed(SEED)
    np.random.seed(SEED)

    # Load graphs

    with open("data/processed/multiplex_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)


    with open("data/processed/train_edges.pkl", "rb") as f:
        train_edges = pickle.load(f)


    # Train negatives: used to train the classifier on negative
    # examples that are NOT the test negatives.

    with open("data/processed/train_negative_edges.pkl", "rb") as f:
        train_negative_edges = pickle.load(f)


    # Validation edges: used ONLY to tune the F1 threshold
    # (leakage-free -- never used to train the classifier).

    with open("data/processed/val_edges.pkl", "rb") as f:
        val_edges = pickle.load(f)


    with open("data/processed/val_negative_edges.pkl", "rb") as f:
        val_negative_edges = pickle.load(f)


    with open("data/processed/test_edges.pkl", "rb") as f:
        test_edges = pickle.load(f)


    with open("data/processed/test_negative_edges.pkl", "rb") as f:
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


        print("Generating Node2Vec embeddings...")


        node2vec = Node2Vec(
            train_graph,
            dimensions=64,
            walk_length=20,
            num_walks=100,
            workers=1,
            seed=SEED
        )


        embedding_model = node2vec.fit(
            window=10,
            min_count=1
        )


        # ==================================
        # TRAIN the classifier on TRAIN edges only
        # (embeddings come only from the train graph, so the
        #  classifier never sees val/test structure)
        # ==================================

        X_train, y_train = build_edge_features(
            embedding_model,
            train_edges[layer_name],
            train_negative_edges[layer_name]
        )


        classifier = LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )

        classifier.fit(X_train, y_train)


        # ==================================
        # VALIDATION: tune the F1 threshold
        # (leakage-free -- test data is never used here)
        # ==================================

        X_val, y_val = build_edge_features(
            embedding_model,
            val_edges[layer_name],
            val_negative_edges[layer_name]
        )

        val_scores = classifier.predict_proba(X_val)[:, 1]

        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_scores
        )

        print(
            f"\nBest threshold (from val): {best_threshold:.4f} "
            f"(val F1 = {best_val_f1:.4f})"
        )


        # ==================================
        # EVALUATE only on held-out TEST edges
        # ==================================

        X_test, y_test = build_edge_features(
            embedding_model,
            test_edges[layer_name],
            test_negative_edges[layer_name]
        )

        scores = classifier.predict_proba(X_test)[:, 1]


        # Save predictions (aligned with X_test order:
        # positives first, then negatives)

        layer_predictions = []

        index = 0


        for edge in test_edges[layer_name]:

            u = edge[0]
            v = edge[1]

            layer_predictions.append(
                (u, v, float(scores[index]), 1, layer_name)
            )

            index += 1


        for u, v in test_negative_edges[layer_name]:

            layer_predictions.append(
                (u, v, float(scores[index]), 0, layer_name)
            )

            index += 1


        all_predictions.extend(layer_predictions)


        # ==================================
        # Evaluation (F1 uses validation-tuned threshold)
        # ==================================

        results = evaluate_all(
            y_test,
            scores,
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


        print("\nEvaluation Results:")
        print(f"ROC-AUC: {results['ROC-AUC']:.4f}")
        print(f"PR-AUC: {results['PR-AUC']:.4f}")
        print(f"F1-score: {results['F1-score']:.4f}")
        print(f"Threshold_Used: {results['Threshold_Used']:.4f}")


    if RUN_TAG:
        # multi-seed run: keep canonical results untouched
        save_metrics(
            all_metrics,
            f"node2vec_{RUN_TAG}.csv",
            base_folder="experiments/multiseed/metrics"
        )
        save_predictions(
            all_predictions,
            f"node2vec_{RUN_TAG}.csv",
            base_folder="experiments/multiseed/predictions"
        )
    else:
        save_metrics(
            all_metrics,
            "embedding/node2vec_metrics.csv"
        )
        save_predictions(
            all_predictions,
            "embedding/node2vec_predictions.csv"
        )


    print("\nNode2Vec baseline completed successfully!")