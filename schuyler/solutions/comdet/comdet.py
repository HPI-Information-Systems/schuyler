import time

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
import networkx as nx


class ComDetSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for ComDet.")
        return None, None

    def test(self, groundtruth, sql_file_path,schema_file_path, model=None):
        start_time = time.time()
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
        print(clusters)
        # communities = {node: {node} for node in graph.nodes}
        # modularity = nx.algorithms.community.quality.modularity
        # max_modularity = -1
        # best_partition = list(communities.values())
        return clusters, time.time()-start_time