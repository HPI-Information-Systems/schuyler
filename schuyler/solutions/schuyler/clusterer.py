import networkx as nx

def louvain_clustering(graph, attribute=None):
    #set attribute to weight
    if attribute is not None:
        for edge in graph.edges:
            edge = edge["edge"]
            edge["weight"] = edge[attribute]
    clusters = nx.community.louvain_communities(graph)
    result = []
    for cluster in clusters:
        result.append([str(table) for table in cluster])
    return result