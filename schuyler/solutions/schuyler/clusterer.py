import networkx as nx

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