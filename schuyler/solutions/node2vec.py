import time
import numpy as np
import os
import torch
import random
from schuyler.solutions.schuyler.utils import get_database_descriptions

from sklearn.cluster import AffinityPropagation
from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from node2vec import Node2Vec

class Node2VecSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, model, prompt_base_path,prompt_model, description_type, seed, sql_file_path=None, schema_file_path=None, iteration=0):
        start_time = time.time()
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        database_descriptions = get_database_descriptions(self.database, prompt_base_path, description_type, iteration)
        G = DatabaseGraph(self.database)
        database_name = self.database.database.split("__")[1]
        cache_folder = f"/data/{database_name}/results/{description_type}_{prompt_model.__name__}/{iteration}"
        G.construct(database_descriptions, cache_folder=cache_folder)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        node2vec = Node2Vec(G.graph, dimensions=64, walk_length=10, num_walks=100, p=1, q=1, workers=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)


        embeddings = [model.wv[str(node)] for node in G.graph.nodes()]
        affinity_propagation = AffinityPropagation()
        labels = affinity_propagation.fit_predict(embeddings)
        cluster_result = []
        for i in range(len(set(labels))):
            cluster_result.append([])
        for i, label in enumerate(labels):
            cluster_result[label].append(G.nodes[i].table.table_name)

        output = []
        for i in range(len(set(labels))):
            output.append([])
        for i, label in enumerate(labels):
            output[label].append(G.nodes[i].table.table_name)
        # print("CLustering node clusterings")
        return output, time.time()-start_time
        # return node_clusterings[0], time.time()-start_time
