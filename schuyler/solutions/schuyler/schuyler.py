import time
import numpy as np
import os
import pickle
import wandb
import pandas as pd
import copy
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
        G = DatabaseGraph(self.database)
        # G.construct(similar_table_connection_threshold, groundtruth=groundtruth)
        
        test_table = self.database.get_tables()[0].get_df()
        print("table", test_table)
        s = TutaModel(database=self.database)
        print('Model loaded')
        start = time.time()
        emb = model.get_embedding(test_table)
        print(f'Embedding generated {time.time()-start}s')
        print(emb)
        end = time.time()
        print(f'tot emb time: {end - start}s')
        print('Done')

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


