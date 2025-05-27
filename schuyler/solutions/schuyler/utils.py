import numpy as np

def normalize_edge_weights(graph, weight_attribute="weight"):
    edge_weights = [graph[edge[0]][edge[1]].get(weight_attribute, 0) for edge in graph.edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    if max_weight == min_weight:
        raise ValueError("All edge weights are identical; normalization is not possible.")
    for edge in graph.edges:
        original_weight = graph[edge[0]][edge[1]].get(weight_attribute, 0)
        normalized_weight = (original_weight - min_weight) / (max_weight - min_weight)
        graph[edge[0]][edge[1]][weight_attribute] = normalized_weight
    return graph

def l2_normalization(embeddings):
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norm, 1e-10)
    return normalized_embeddings

