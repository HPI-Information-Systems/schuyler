import time
import numpy as np
import wandb
import pandas as pd
import torch
import os
import random
from sklearn.metrics import silhouette_score

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.utils import get_database_descriptions

class SchuylerSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, prompt_model, finetune,triplet_generation_model, clustering_method, model, prompt_base_path, description_type, sql_file_path=None, schema_file_path=None, seed=42, iteration=0):
        start_time = time.time()
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        start = time.time()
        database_descriptions = get_database_descriptions(self.database, prompt_base_path, description_type, iteration)
        wandb.log({"description_time": time.time() - start})
        G = DatabaseGraph(self.database)
        start = time.time()
        database_name = self.database.database.split("__")[1]
        cache_folder = f"/data/{database_name}/results/{description_type}_{prompt_model.__name__}/{iteration}"
        G.construct(database_descriptions, cache_folder=cache_folder)
        wandb.log({"graph_construction_time": time.time() - start})
        
        sim_matrix_path = os.path.join(cache_folder, "sim_matrix.csv")
        sim_matrix = pd.read_csv(sim_matrix_path, index_col=0, header=0)
        if hasattr(G, 'model'):
            if hasattr(G.model, 'cleanup'):
                G.model.cleanup()
        start = time.time()
        tm = triplet_generation_model(self.database, G, sim_matrix)
        triplets = tm.generate_triplets()
        wandb.log({"triplet_generation_time": time.time() - start})
        if finetune:
            start = time.time()
            G.model.finetune(triplets, tm, seed=seed)
            wandb.log({"finetuning_time": time.time() - start})
            G.update_encodings()
        node_clusterings = []
        features = []
        tables = {}
        start = time.time()
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.encoding)
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
        wandb.log({"clustering_time": time.time() - start})
        result = []
        for i in range(len(set(labels))):
            result.append([])
        for i, label in enumerate(labels):
            result[label].append(tables[i])
        node_clusterings.append(result)
        output = node_clusterings[0]
        return output, time.time()-start_time

def determine_k(X, clustering_method):
    k_range = range(2, 20)
    scores = []
    for k in k_range:
        c = clustering_method(k)
        c.fit(X)
        labels = c.predict(X) if hasattr(c, "predict") else c.labels_
        shilouette_score = silhouette_score(X, labels)
        scores.append(shilouette_score)
    optimal_k_silhouette = k_range[scores.index(max(scores))]
    return optimal_k_silhouette