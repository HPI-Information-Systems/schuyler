import networkx as nx
from networkx.algorithms.centrality import edge_betweenness_centrality
from scipy.sparse.linalg import eigsh
import numpy as np

from schuyler.solutions.iDisc.base_clusterer.BaseClusterer import BaseClusterer
from schuyler.solutions.iDisc.preprocessor import GraphRepresentator

class LinkClust(BaseClusterer):
    def __init__(self, representator, del_method='betweenness'):
        super().__init__()
        self.del_method = del_method
        self.representator = representator

    def cluster(self, tables):
        g: nx.Graph = self.representator.get_representation()
        max_q = float('-inf')
        best_clusters = None
        while g.number_of_edges() > 0:
            print(f"Edges: {g.number_of_edges()}")
            clusters = list(nx.connected_components(g))
            q = self.cluster_quality(g,clusters)
            edge_to_del = self.edge_del(g, self.del_method)
            if edge_to_del is None:
                break
            g.remove_edge(*edge_to_del)
            print(f"New cluster: {clusters}, q: {q}")
            if q > max_q:
                max_q = q
                best_clusters = clusters
        print(f"Best clusters: {best_clusters}")
        return best_clusters

    def cluster_quality(self, g, clusters):
        no_of_edges = g.number_of_edges()
        def q_prime(cluster):
            nodes = set(cluster)
            internal_edges = sum(1 for u, v in g.edges if u in nodes and v in nodes)
            incident_edges = sum(1 for u, v in g.edges if u in nodes or v in nodes)
            prob_e_to_ci = internal_edges / no_of_edges
            exp_prob = (incident_edges / no_of_edges) ** 2
            return prob_e_to_ci - exp_prob
        return sum(q_prime(cluster) for cluster in clusters)
        
    def edge_del(self, g: nx.Graph, method='betweenness'):
        if method == 'betweenness':
            return self._remove_edge_based_on_betweenness(g)
        elif method == 'spectral_graph':
            return self.spectral_graph_partitioning(g)
        else:
            raise ValueError("Invalid method")

    def _remove_edge_based_on_betweenness(self, graph):
        betweenness = edge_betweenness_centrality(graph)
        return max(betweenness, key=betweenness.get)
    
    def spectral_graph_partitioning(self, graph):
        conn_comps = list(nx.connected_components(graph))
        conn_comp = max(conn_comps, key=len)
        graph = graph.subgraph(conn_comp)
        tables = np.array(graph.nodes)
        print(tables)
        laplacian = nx.laplacian_matrix(graph).astype(float)
        num_nodes = laplacian.shape[0]
        print(f"Number of nodes: {num_nodes}")
        if num_nodes <= 2:
            print("Insufficient nodes for spectral partitioning.")
            return None
        _, eigenvectors = eigsh(laplacian, k=2, which='SM')
        fiedler_vector = eigenvectors[:, 1]
        clusters = fiedler_vector > 0
        cluster1 = tables[clusters]
        cluster2 = tables[~clusters]
        inter_cluster_edges = [
            (u, v) for u, v in graph.edges if (u in cluster1 and v in cluster2) or (u in cluster2 and v in cluster1)]
        print(f"Inter cluster edges: {inter_cluster_edges}")
        #todo check if this really correct
        return inter_cluster_edges[0]