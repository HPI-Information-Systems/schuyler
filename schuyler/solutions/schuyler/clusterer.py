import networkx as nx
from sklearn.cluster import AffinityPropagation
import numpy as np
from igraph import Graph
import sys
import networkx as nx
import numpy as np
import sys
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA

def louvain_clustering(graph, attribute=None):
    #set attribute to weight
    if attribute is not None:
        for edge in graph.edges:
            edge = graph[edge[0]][edge[1]]["edge"] 
            setattr(edge, 'weight', getattr(edge, attribute))
            #edge.weight = edge[attribute]
    clusters = nx.community.louvain_communities(graph)
    result = []
    for cluster in clusters:
        result.append([str(table) for table in cluster])
    return result
def leiden_clustering(graph, attribute=None):
    ig_graph = Graph.from_networkx(graph)
    if attribute is not None:
        if attribute not in ig_graph.es.attributes():
            raise ValueError(f"Attribute '{attribute}' not found in graph edges.")
        ig_graph.es["weight"] = ig_graph.es[attribute]
    partition = ig_graph.community_infomap(edge_weights="weight" if attribute else None)
    result = []
    for cluster in partition:
        # for node in cluster:
        #     print("nod", node)
        #     print(ig_graph.vs[node])
        result.append([str(ig_graph.vs[node]["_nx_name"]) for node in cluster])
    return result

def affinity_propagation_clustering(graph, attribute=None):
    if attribute is not None:
        for edge in graph.edges:
            weight = graph[edge[0]][edge[1]].get(attribute, 1.0)
            graph[edge[0]][edge[1]]["weight"] = weight

    adjacency_matrix = nx.to_numpy_array(graph, weight="weight")
    np.set_printoptions(threshold=sys.maxsize)
    affinity_propagation = AffinityPropagation(affinity='precomputed',damping=0.9, random_state=0)
    affinity_propagation.fit(adjacency_matrix)

    cluster_labels = affinity_propagation.labels_
    
    nodes = list(graph.nodes)
    clusters = {}
    for node, label in zip(nodes, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(str(node))
    
    return list(clusters.values())

def affinity_propagation_clustering_with_pca(graph, attribute=None, n_components=20):
    if attribute is not None:
        for edge in graph.edges:
            weight = graph[edge[0]][edge[1]].get(attribute, 1.0)
            graph[edge[0]][edge[1]]["weight"] = weight

    adjacency_matrix = nx.to_numpy_array(graph, weight="weight")
    np.set_printoptions(threshold=sys.maxsize)
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(adjacency_matrix)

    affinity_propagation = AffinityPropagation(affinity='precomputed', damping=0.9, random_state=0)
    similarity_matrix = reduced_matrix @ reduced_matrix.T
    affinity_propagation.fit(similarity_matrix)
    cluster_labels = affinity_propagation.labels_
    nodes = list(graph.nodes)
    clusters = {}
    for node, label in zip(nodes, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(str(node))
    
    return list(clusters.values())