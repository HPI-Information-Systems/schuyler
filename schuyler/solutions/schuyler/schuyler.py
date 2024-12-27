import time
import numpy as np
import networkx as nx
from sklearn import metrics
from collections import defaultdict
from sklearn.cluster import AffinityPropagation, spectral_clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

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

    def test(self, model):
        start_time = time.time()
        G = DatabaseGraph(self.database)
        G.construct()
        edge_clusterings = []
        

        louvain_1 = louvain_clustering(G.graph, "weight")
        edge_clusterings.append([])

        node_clusterings = []
        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.features)
        X = np.array(features) 

        ap = AffinityPropagation(damping=0.9)
        labels = ap.fit_predict(X)
        result = []
        for i in range(len(set(labels))):
            result.append([])
        for i, label in enumerate(labels):
            result[label].append(tables[i])
        node_clusterings.append(result)

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
        cluster = MetaClusterer(G.graph).cluster(node_clusterings)
        cluster = MetaClusterer(G.graph).cluster([cluster, clusterings[0]])
        print("labels", labels)
        
        print(cluster)
        return cluster, time.time()-start_time


        
