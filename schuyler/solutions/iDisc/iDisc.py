import time

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
import matplotlib.pyplot as plt
import networkx as nx

from schuyler.solutions.iDisc.base_clusterer import SimilarityBasedClusterer, LinkClust
from schuyler.solutions.iDisc.trees import HierarchicalClusterTree, ClusterTree

class iDiscSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__
        print("NOT IMPLEMENTED: CHECK THAT TWICE THE SAME CLUSTERER / REPRESENTATOR IS NOT USED")

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for iDisc.")
        return None, None

    def test(self, tree, sim_clusterers, link_clusterers, groundtruth, model=None):
        start_time = time.time()
        base_clusterers = []
        for sim_cluster in sim_clusterers:
            representator = sim_cluster["representator"]["module"](database=self.database, **sim_cluster["representator"]["params"])
            base_clusterers.append(SimilarityBasedClusterer(representator, sim_cluster["linkage"], sim_cluster["metric"]))
        for link_cluster in link_clusterers:
            representator = link_cluster["representator"]["module"](database=self.database, **link_cluster["representator"]["params"])
            base_clusterers.append(LinkClust(representator, link_cluster["del_method"]))
        print(base_clusterers)
        self.tree = tree(self.database)
        self.tree.construct_tree(base_clusterers)
        clusters = self.tree.cluster()
        return clusters, time.time()-start_time


        
