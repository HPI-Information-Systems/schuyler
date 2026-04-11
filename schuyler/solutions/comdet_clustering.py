import numpy as np
import time
from sklearn.cluster import AffinityPropagation
import networkx as nx

from schuyler.database.database import Database
from schuyler.solutions.schuyler.utils import get_database_descriptions
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import leiden_clustering

class ComDetClusteringSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for ComDet.")
        return None, None

    def test(self,prompt_base_path,seed,prompt_model, description_type, sql_file_path=None,schema_file_path=None, model=None, iteration=0):
        start_time = time.time()
        database_descriptions = get_database_descriptions(self.database, prompt_base_path, description_type, iteration)
        G = DatabaseGraph(self.database)
        database_name = self.database.database.split("__")[1]
        cache_folder = f"/data/{database_name}/results/{description_type}_{prompt_model.__name__}/{iteration}"
        G.construct(database_descriptions, cache_folder=cache_folder)
        clustering = leiden_clustering(G.graph, "weight")
        tables = self.database.get_tables()
        table_names = [table.table_name for table in tables]
        graph = nx.Graph()
        graph.add_nodes_from(table_names)
        for table in tables:
            fks = table.get_foreign_keys()
            for fk in fks:
                graph.add_edge(fk["constrained_table"], fk["referred_table"])
        clusters = nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)
        clusters = [list(cluster) for cluster in clusters]
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
        clusters = MetaClusterer(G.graph).cluster([cluster_result, clustering], 0.5)
        return clusters, time.time()-start_time