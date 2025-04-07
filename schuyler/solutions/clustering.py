import pandas as pd
import numpy as np

import time

from schuyler.database.database import Database
from sklearn.cluster import AffinityPropagation

from schuyler.solutions.base_solution import BaseSolution

from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering, affinity_propagation_clustering,leiden_clustering,affinity_propagation_clustering_with_pca

class ClusteringSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for ComDet.")
        return None, None

    def test(self, groundtruth, prompt_base_path,similar_table_connection_threshold,description_type, sql_file_path=None,schema_file_path=None, model=None):
        start_time = time.time()
        G = DatabaseGraph(self.database)
        G.construct(prompt_base_path=prompt_base_path, description_type=description_type,similar_table_connection_threshold=similar_table_connection_threshold, groundtruth=groundtruth)
        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.embeddings)
        X = np.array(features) 
        ap = AffinityPropagation()
        labels = ap.fit_predict(X)
        cluster_result = []
        for i in range(len(set(labels))):
            cluster_result.append([])
        for i, label in enumerate(labels):
            cluster_result[label].append(tables[i])
        return cluster_result, time.time()-start_time