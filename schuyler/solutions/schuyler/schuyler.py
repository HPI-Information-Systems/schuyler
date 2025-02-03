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

    def test(self, no_of_hierarchy_levels,min_max_normalization_sim_matrix, finetune,triplet_generation_model, similar_table_connection_threshold, model, groundtruth=None):
        start_time = time.time()
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42)
        np.random.seed(42)
        G = DatabaseGraph(self.database, TutaModel)
        G.construct(similar_table_connection_threshold, groundtruth=groundtruth)
        database_name = self.database.database.split("__")[1]
        sim_matrix = pd.read_csv(f"/data/{database_name}/sim_matrix.csv", index_col=0, header=0)
        if min_max_normalization_sim_matrix:
            sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min())
        start = time.time()
        tm = triplet_generation_model(self.database, G, sim_matrix, groundtruth)
        triplets = tm.generate_triplets()
        print(f'Embedding generated {time.time()-start}s')
        if finetune:
            G.visualize_embeddings(name="before_finetuning")
            G.model.finetune(triplets, tm)
            print()
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
        ap = AffinityPropagation(damping=0.9)
        labels = ap.fit_predict(X)
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


