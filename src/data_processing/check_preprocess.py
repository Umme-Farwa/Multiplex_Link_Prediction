import pickle


with open(
    "data/processed/multiplex_graphs.pkl",
    "rb"
) as f:
    graphs = pickle.load(f)


print("Layers:")
print(list(graphs.keys()))


print("\nLayer Statistics:")

for name, graph in graphs.items():

    print(
        name,
        "| Nodes:",
        graph.number_of_nodes(),
        "| Edges:",
        graph.number_of_edges()
    )