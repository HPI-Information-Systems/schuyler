import time
import numpy as np
import os
import pickle
import networkx as nx
from sklearn.cluster import AffinityPropagation


from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering, affinity_propagation_clustering,leiden_clustering,affinity_propagation_clustering_with_pca,girvan_newman_clustering
from schuyler.solutions.schuyler.graph import Edge, Node

class SchuylerSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, no_of_hierarchy_levels, model):
        start_time = time.time()
        features = []
        tables = {}
        G = DatabaseGraph(self.database)
        G.construct()
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.embeddings)
        X = np.array(features) 
        ap = AffinityPropagation()
        labels = ap.fit_predict(X)
        base_clustering = []
        for i in range(len(set(labels))):
            base_clustering.append([])
        for i, label in enumerate(labels):
            base_clustering[label].append(tables[i])

        
        nodes = G.graph.nodes
        clustered_graph = nx.Graph()
        nodes = [Node(table, llm=G.llm, st=G.sentencetransformer) for table in self.database.get_tables()]
        clustered_graph.add_nodes_from(nodes)
        tables = [node.table for node in G.graph.nodes]
        for table in tables:
            for fk in table.get_foreign_keys():
                table1 = table.table_name
                table2 = fk["referred_table"]
                #check if table1 and table2 are in a different cluster and not the same!
                if table1 != table2 and any(table1 in cluster and table2 in cluster for cluster in base_clustering):

                    edge = Edge(get_node(clustered_graph.nodes, table1), get_node(clustered_graph.nodes, table1), G.sentencetransformer)
                    clustered_graph.add_edge(edge.node1, edge.node2)
                    clustered_graph[edge.node1][edge.node2]["edge"] = edge
                    clustered_graph[edge.node1][edge.node2]["weight"] = 1.0 #* 100
        for cluster in base_clustering:
            for table1 in cluster:
                for table2 in cluster:
                    if table1 == table2:
                        continue
                    edge = Edge(get_node(clustered_graph.nodes, table1), get_node(clustered_graph.nodes, table2), G.sentencetransformer)
                    clustered_graph.add_edge(edge.node1, edge.node2)
                    clustered_graph[edge.node1][edge.node2]["edge"] = edge
                    clustered_graph[edge.node1][edge.node2]["weight"] = edge.table_sim / 2
        
        print()
        print("Graph constructed")
        edge_clusterings = []
        clustered_graph = normalize_edge_weights(clustered_graph, weight_attribute="weight")
        # for edge in clustered_graph.edges:
        #     print(edge)
        #     print(clustered_graph[edge[0]][edge[1]]["weight"])
        A = nx.to_numpy_array(clustered_graph)#,# nodelist=clustered_graph.nodes())
        np.savetxt("/data/adjacency_matrix.csv", A, delimiter=",", fmt="%d")
        print("Louvain")
        #print(G.graph.edges)
        louvain_1 = affinity_propagation_clustering(clustered_graph, "weight")
        edge_clusterings.append(louvain_1)
        print("Louvain finished")
        print("FINAL CLUSTERING", louvain_1)
        return louvain_1, time.time()-start_time

def get_node(nodes, table_name):
    for n in nodes:
        if n.table.table_name == table_name:
            return n
    return None

def apply_softmax_to_edge_weights(graph, weight_attribute="weight"):
    edge_weights = [graph[edge[0]][edge[1]].get(weight_attribute, 1.0) for edge in graph.edges]
    exp_weights = np.exp(edge_weights)
    softmax_weights = exp_weights / np.sum(exp_weights)
    for edge, softmax_weight in zip(graph.edges, softmax_weights):
        graph[edge[0]][edge[1]][weight_attribute] = softmax_weight    
    return graph

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


