import time
import numpy as np
import os
import pickle

import wandb
import pandas as pd
import copy
import torch
import random
from datasets import Dataset, DatasetDict
from schuyler.solutions.schuyler.edge import Edge

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import hashlib
from sentence_transformers import util, SentenceTransformer, InputExample, SentencesDataset
import networkx as nx

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering, affinity_propagation_clustering,leiden_clustering,affinity_propagation_clustering_with_pca

from schuyler.solutions.schuyler.utils import normalize_edge_weights
from schuyler.solutions.schuyler.tripletloss import generate_triplets, generate_similar_and_nonsimilar_triplets,generate_triplets_with_groundtruth
from schuyler.solutions.schuyler.triplet_generator.constrained_triplet_generator import ConstrainedTripletGenerator
from schuyler.analyzer.see_foreign_key_cluster_belongness import foreign_key_cluster_belongness


from schuyler.solutions.schuyler.feature_vector.llm import TutaModel
class SchuylerSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, no_of_hierarchy_levels,prompt_model, min_max_normalization_sim_matrix, finetune,triplet_generation_model, clustering_method, similar_table_connection_threshold, model, prompt_base_path, description_type, groundtruth=None, sql_file_path=None, schema_file_path=None, seed=42):
        
        start_time = time.time()


        #! wieder einkommentieren
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        # np.random.seed(seed)

        # G = DatabaseGraph(self.database, TutaModel)
        G = DatabaseGraph(self.database)


        G.construct(prompt_base_path=prompt_base_path,prompt_model=prompt_model,description_type=description_type,similar_table_connection_threshold=similar_table_connection_threshold, groundtruth=groundtruth)
        database_name = self.database.database.split("__")[1]
        folder = f"{description_type}_gpt" if prompt_model == "ChatGPT" else description_type
        sim_matrix_path = f"/data/{database_name}/results/{folder}/sim_matrix.csv"# if 
        # sim_matrix = pd.read_csv(f"/data/{database_name}/sim_matrix.csv", index_col=0, header=0)
        sim_matrix = pd.read_csv(sim_matrix_path, index_col=0, header=0)
        if min_max_normalization_sim_matrix:
            sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min())
        start = time.time()
        tm = triplet_generation_model(self.database, G, sim_matrix, groundtruth)
        triplets = tm.generate_triplets()
        print(f'Embedding generated {time.time()-start}s')
        if finetune:
            G.visualize_embeddings(name="before_finetuning")
            G.model.finetune(triplets, tm, seed=seed)
            # print(G.graph.nodes[0].embeddings)
            G.update_encodings()
            G.visualize_embeddings(name="after_finetuning")
        end = time.time()
        print(f'tot emb time: {end - start}s')
        print('Done')
        node_clusterings = []

        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.encoding)

        X = np.array(features) 
        # ap = AffinityPropagation()
        if clustering_method.__name__ in ["AffinityPropagation", "DBSCAN", "OPTICS"]:
            clustering_method = clustering_method()
        elif clustering_method.__name__ == "GaussianMixture":
            k = determine_k(X, clustering_method)
            print("Optimal K:", k)
            clustering_method = clustering_method(n_components=k)
        else:
            k = determine_k(X, clustering_method)
            print("Optimal K:", k)
            clustering_method = clustering_method(n_clusters=k)
        



        labels = clustering_method.fit_predict(X)

        result = []
        for i in range(len(set(labels))):
            result.append([])
        for i, label in enumerate(labels):
            result[label].append(tables[i])

        node_clusterings.append(result)
        output = node_clusterings[0]
        print("Output", output)
        return output, time.time()-start_time
        # return node_clusterings[0], time.time()-start_time


def calculate_cluster_embedding(cluster, graph):
    encs = [graph.get_node(table).encoding.cpu().numpy() for table in cluster]
    if len(encs) == 1:
        return encs[0]
    return np.mean(encs, axis=0)

def select_most_similar_cluster(cluster_embedding, reference_cluster_embeddings):
    similarities = []
    for i, ref_cluster_embedding in enumerate(reference_cluster_embeddings):
        if not np.array_equal(cluster_embedding, ref_cluster_embedding):
            similarities.append(util.cos_sim(cluster_embedding, ref_cluster_embedding).item())
        else:
            similarities.append(0.0)
    return np.argmax(similarities)
    

def apply_softmax_to_edge_weights(graph, weight_attribute="weight"):
    edge_weights = [graph[edge[0]][edge[1]].get(weight_attribute, 1.0) for edge in graph.edges]
    exp_weights = np.exp(edge_weights)
    softmax_weights = exp_weights / np.sum(exp_weights)
    for edge, softmax_weight in zip(graph.edges, softmax_weights):
        graph[edge[0]][edge[1]][weight_attribute] = softmax_weight    
    return graph


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




