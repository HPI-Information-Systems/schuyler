import time
import numpy as np
import os
import pickle
import pandas as pd
import copy
from datasets import Dataset, DatasetDict
from schuyler.solutions.schuyler.edge import Edge

from sklearn.cluster import AffinityPropagation
import hashlib
from sentence_transformers import util, SentenceTransformer, InputExample, SentencesDataset


from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering, affinity_propagation_clustering,leiden_clustering,affinity_propagation_clustering_with_pca
from schuyler.solutions.schuyler.utils import normalize_edge_weights
from schuyler.solutions.schuyler.tripletloss import generate_triplets, generate_similar_and_nonsimilar_triplets,generate_triplets_with_groundtruth

class SchuylerSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, no_of_hierarchy_levels, similar_table_connection_threshold, model, groundtruth=None):
        start_time = time.time()
        sim_matrix = pd.read_csv("/data/magento/sim_matrix.csv", index_col=0, header=0)
        sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min())
        
        G = DatabaseGraph(self.database)
        G.construct(similar_table_connection_threshold, groundtruth=groundtruth)
        #add edges that are above a threshold
        threshold = 0.7
        for i, row in sim_matrix.iterrows():
            for j, value in row.items():
                print(i, j, value)
                if value > threshold:
                    edge = Edge(G.get_node(i), G.get_node(j), G.sentencetransformer, sim=value)
                    G.graph.add_edge(edge.node1, edge.node2)
                    G.graph[edge.node1][edge.node2]["edge"] = edge
                    # if use_tfidf:
                    #     self.graph[edge.node1][edge.node2]["weight"] = tfidf_sim.loc[table.table_name, fk["referred_table"]]
                    # else:
                    G.graph[edge.node1][edge.node2]["weight"] = edge.table_sim


        #raise ValueError
        print("Graph constructed")
        node_descriptions = {str(node): node.llm_description for node in G.graph.nodes}
        print("Node descriptions", node_descriptions)

        #convert to float
        #sim_matrix = sim_matrix.apply(pd.to_numeric, errors='coerce')
        #normalize sim matrix (min-max)


        # triplets = generate_triplets(G.graph, G.sentencetransformer, num_triplets_per_anchor=5, similarity_threshold=0.5)
        triplets = generate_similar_and_nonsimilar_triplets(G.graph, sim_matrix, num_triplets_per_anchor=5, high_similarity_threshold=0.7, low_similarity_threshold=0.2)
        # triplets = generate_triplets_with_groundtruth(G.graph, groundtruth.clusters)
        print("Triplets generated", triplets)
        # train_examples = [
        #     InputExample(texts=[node_descriptions[anchor], node_descriptions[positive], node_descriptions[negative]])
        #     for anchor, positive, negative in triplets
        # ]
        # print("Train examples", train_examples)

        # train_dataset = SentencesDataset(train_examples, model=model)
        print(node_descriptions)
        anchors = [node_descriptions[anchor] for anchor, _, _ in triplets]
        positives = [node_descriptions[positive] for _, positive, _ in triplets]
        negatives = [node_descriptions[negative] for _, _, negative in triplets]

        data = {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        }

        # Create a Dataset object
        train_dataset = Dataset.from_dict(data)
        G.visualize_embeddings(name="before_finetuning")
        G.sentencetransformer.finetune(train_dataset, 20, 100)
        print()
        # print(G.graph.nodes[0].embeddings)
        G.update_encodings()
        G.visualize_embeddings(name="after_finetuning")
        # print(G.graph.nodes[0].embeddings)
        # print("Sim ", util.cos_sim(G.graph.nodes[0].embeddings, x))

        # print("Graph constructed")
        # edge_clusterings = []
        # G.graph = normalize_edge_weights(G.graph, weight_attribute="weight")
        # # print("Louvain")
        # # print(G.graph.edges)
        # louvain_1 = leiden_clustering(G.graph, "weight")
        # edge_clusterings.append(louvain_1)
        # print("Louvain finished")

        node_clusterings = []
        # # features = []
        # # tables = {}
        # # print("Calculating features")
        # # for i, node in enumerate(G.graph.nodes):
        # #     tables[i] = node.table.table_name
        # #     features.append(node.features)
        # # X = np.array(features) 
        # # print("Features calculated")
        # # print("Affinity Propagation")
        # # ap = AffinityPropagation()
        # # labels = ap.fit_predict(X)
        # # result = []
        # # for i in range(len(set(labels))):
        # #     result.append([])
        # # for i, label in enumerate(labels):
        # #     result[label].append(tables[i])
        # # node_clusterings.append(result)

        features = []
        tables = {}
        for i, node in enumerate(G.graph.nodes):
            tables[i] = node.table.table_name
            features.append(node.encoding)
        X = np.array(features) 

        ap = AffinityPropagation()
        labels = ap.fit_predict(X)
        result = []
        for i in range(len(set(labels))):
            result.append([])
        for i, label in enumerate(labels):
            result[label].append(tables[i])
        node_clusterings.append(result)

        # print("CLustering node clusterings")
        # #cluster = MetaClusterer(G.graph).cluster(node_clusterings)
        # print("Merging edge and node clus^ter")
        # # return node_clusterings[0], time.time()-start_time
        # #print(edge_clusterings[0])
        # # return edge_clusterings[0], time.time()-start_time
        # clustering = MetaClusterer(G.graph).cluster([node_clusterings[0], edge_clusterings[0]], 0.66)
        # print("Preliminary clustering", clustering)


        # save to file
        # with open("/data/clustering.pkl", "wb") as f:
        #     pickle.dump(clustering, f)
        #load clustering object
        # with open("/data/clustering.pkl", "rb") as f:
        #     clustering = pickle.load(f)
        # return clustering, time.time()-start_time
        
        # print("labels", labels)
        
        #print(cluster)
        print(node_clusterings[0])
        return node_clusterings[0], time.time()-start_time

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


