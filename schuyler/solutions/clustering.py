import pandas as pd
import numpy as np

import time

from schuyler.database.database import Database
from schuyler.solutions.schuyler.schuyler import determine_k
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.utils import get_database_descriptions

from schuyler.solutions.schuyler.graph import DatabaseGraph

class ClusteringSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for Clustering.")
        return None, None

    def test(self,prompt_model,seed,clustering_method, prompt_base_path,description_type, sql_file_path=None,schema_file_path=None, model=None, iteration=0):
        start_time = time.time()
        database_descriptions = get_database_descriptions(self.database, prompt_base_path, description_type, iteration=iteration)
        G = DatabaseGraph(self.database)
        database_name = self.database.database.split("__")[1]
        cache_folder = f"/data/{database_name}/results/{description_type}_{prompt_model.__name__}/{iteration}"
        G.construct(database_descriptions, cache_folder=cache_folder)
        
        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.embeddings)
        X = np.array(features) 
        if clustering_method.__name__ in ["AffinityPropagation", "DBSCAN", "OPTICS"]:
            clustering_method = clustering_method()
        elif clustering_method.__name__ == "GaussianMixture":
            k = determine_k(X, clustering_method)
            clustering_method = clustering_method(n_components=k)
        else:
            k = determine_k(X, clustering_method)
            clustering_method = clustering_method(n_clusters=k)
        
        labels = clustering_method.fit_predict(X)
        cluster_result = []
        for i in range(len(set(labels))):
            cluster_result.append([])
        for i, label in enumerate(labels):
            cluster_result[label].append(tables[i])
        return cluster_result, time.time()-start_time