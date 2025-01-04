import time
import numpy as np
import os
import pickle
from sklearn.cluster import AffinityPropagation

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering

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
        
        G = DatabaseGraph(self.database)
        G.construct()

        print("Graph constructed")
        edge_clusterings = []
        print("Louvain")
        print(G.graph.edges)
        louvain_1 = louvain_clustering(G.graph, "weight")
        edge_clusterings.append(louvain_1)
        print("Louvain finished")

        node_clusterings = []
        # features = []
        # tables = {}
        # print("Calculating features")
        # for i, node in enumerate(G.graph.nodes):
        #     tables[i] = node.table.table_name
        #     features.append(node.features)
        # X = np.array(features) 
        # print("Features calculated")
        # print("Affinity Propagation")
        # ap = AffinityPropagation(damping=0.9)
        # labels = ap.fit_predict(X)
        # result = []
        # for i in range(len(set(labels))):
        #     result.append([])
        # for i, label in enumerate(labels):
        #     result[label].append(tables[i])
        # node_clusterings.append(result)

        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.embeddings)
        X = np.array(features) 

        ap = AffinityPropagation(damping=0.9)
        labels = ap.fit_predict(X)
        result = []
        for i in range(len(set(labels))):
            result.append([])
        for i, label in enumerate(labels):
            result[label].append(tables[i])
        node_clusterings.append(result)
        print("CLustering node clusterings")
        #cluster = MetaClusterer(G.graph).cluster(node_clusterings)
        print("Merging edge and node cluster")
        return node_clusterings[0], time.time()-start_time
        cluster = MetaClusterer(G.graph).cluster([node_clusterings[0]])#, edge_clusterings[0]])
        print("labels", labels)
        
        print(cluster)
        return cluster, time.time()-start_time


        
