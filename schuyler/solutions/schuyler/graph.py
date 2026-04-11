from schuyler.database.database import Database
from networkx import Graph, pagerank, betweenness_centrality
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.feature_vector.llm import SentenceTransformerModel
import gc
import time
import pandas as pd
import wandb
import torch
import os
from tqdm import tqdm
import sys
import pickle
class DatabaseGraph:
    def __init__(self, database: Database, model=SentenceTransformerModel, triplet_model=None):
        self.graph = Graph()
        self.database = database
        self.model = model(database)

    def construct(self, database_descriptions, cache_folder=None):
        print("Constructing graph")
        self.nodes = []
        start_time = time.time()
        self.nodes = [Node(table,database_descriptions[table.table_name], model=self.model) for table in self.database.get_tables()]
        wandb.log({"node_creation_time": time.time() - start_time})
        self.graph.add_nodes_from(self.nodes)
        for node1 in self.nodes:
            table = node1.table
            for fk in table.get_foreign_keys():
                edge = Edge(node1, self.get_node(fk["referred_table"]), self.model)
                if edge.table_sim < 0.5:
                    continue
                self.graph.add_edge(edge.node1, edge.node2)
                self.graph[edge.node1][edge.node2]["edge"] = edge
                self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
        pr = pagerank(self.graph, alpha=0.85)
        b = betweenness_centrality(self.graph, normalized=True, endpoints=False, seed=42)
        for node in tqdm(self.nodes, file=sys.stdout):
            path = os.path.join(cache_folder, "nodes")
            node_features_file = f"{path}/{node.table.table_name}.pkl"
            os.makedirs(path, exist_ok=True)
            if os.path.exists(node_features_file):
                with open(node_features_file, "rb") as f:
                    node_features = pickle.load(f)
                node.page_rank = node_features["page_rank"]
                node.degree = node_features["degree"]
                node.betweenness_centrality = node_features["betweenness_centrality"]
                node.embeddings = node_features["embeddings"]
                node.features = node_features["features"]
                node.average_semantic_similarity = node_features["features"][3]
                node.amount_of_fks = node_features["features"][4]
                node.amount_of_columns = node_features["features"][5]
                node.row_count = node_features["features"][6]
            else:
                node.page_rank = pr[node]
                node.degree = self.graph.degree(node)
                node.betweenness_centrality = b[node]
                v = node.calculate_feature_vector(self.graph)
                node.embeddings = v["embeddings"].tolist()
                node.average_semantic_similarity = v["average_semantic_similarity"]
                node.amount_of_fks = v["amount_of_fks"]
                node.amount_of_columns = v["amount_of_columns"]
                node.row_count = v["row_count"]
                node.features = [node.page_rank, node.degree, node.betweenness_centrality, v["average_semantic_similarity"], v["amount_of_fks"], v["amount_of_columns"], v["row_count"]]
                with open(node_features_file, "wb") as f:
                    pickle.dump({"page_rank": node.page_rank, "degree": node.degree, "betweenness_centrality": node.betweenness_centrality, "embeddings": node.embeddings, "features": node.features}, f)
        self.get_similar_embedding_tables(cache_folder, 0.0)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        return self.graph
    
    
    def update_encodings(self):
        for node in self.nodes:
            node.update_encoding(self.model)
        for edge in self.graph.edges:
            edge = self.graph[self.get_node(str(edge[0]))][self.get_node(str(edge[1]))]["edge"] 
            self.graph[edge.node1][edge.node2]["weight"] = edge.get_table_similarity()
            self.graph[edge.node1][edge.node2]["edge"].sim = edge.get_table_similarity()

    def get_node(self, table_name):
        if type(table_name) == Node:
            return table_name
        for n in self.graph.nodes:
            if n.table.table_name == table_name:
                return n
        return None
    
    def get_similar_embedding_tables(self, folder, threshold):
        tables = self.database.get_tables()
        sim_matrix = pd.DataFrame(index=[t.table_name for t in tables], columns=[t.table_name for t in tables])
        similar_tables = []
        sim_matrix_path = f"{folder}/sim_matrix.csv"
        if not os.path.exists(sim_matrix_path):
            for table1 in tqdm(tables):
                table1 = table1.table_name
                for table2 in tables:
                    table2 = table2.table_name
                    if table1 == table2:
                        continue
                    edge = Edge(self.get_node(table1), self.get_node(table2), self.model)
                    sim = edge.table_sim
                    sim_matrix.loc[table1, table2] = sim
                    if sim > threshold:
                        similar_tables.append((table1, table2))
        else:
            sim_matrix = pd.read_csv(sim_matrix_path, index_col=0)
            for table1 in tables:
                table1 = table1.table_name
                for table2 in tables:
                    table2 = table2.table_name
                    if table1 == table2:
                        continue
                    sim = sim_matrix.loc[table1, table2]
                    if sim > threshold:
                        similar_tables.append((table1, table2, sim))
        sim_matrix.to_csv(sim_matrix_path)
        return similar_tables