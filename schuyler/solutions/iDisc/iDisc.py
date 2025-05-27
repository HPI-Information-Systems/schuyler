import time

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
import matplotlib.pyplot as plt
import networkx as nx

from schuyler.solutions.iDisc.base_clusterer import SimilarityBasedClusterer, LinkClust
from schuyler.solutions.iDisc.trees import HierarchicalClusterTree, ClusterTree
from schuyler.solutions.iDisc.preprocessor import VectorRepresentator, SimilarityBasedRepresentator, GraphRepresentator
from schuyler.solutions.iDisc.meta_clusterer.meta_clusterer import MetaClusterer
from schuyler.solutions.iDisc.preprocessor.document_builder import TableNameAndColsDocumentBuilder

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
        vec_rep = VectorRepresentator(database=self.database, document_builder=TableNameAndColsDocumentBuilder)#!
        sim_rep = SimilarityBasedRepresentator(database=self.database)#!
        graph_rep = GraphRepresentator(database=self.database)#!

        meta_clusterer = MetaClusterer(self.database)

        vec_sl = SimilarityBasedClusterer(vec_rep, "single").cluster(self.database.get_tables())
        vec_al = SimilarityBasedClusterer(vec_rep, "average").cluster(self.database.get_tables())
        vec_cl = SimilarityBasedClusterer(vec_rep, "complete").cluster(self.database.get_tables())
        vec_cluster = meta_clusterer.cluster([vec_sl, vec_al, vec_cl])


        sim_sl = SimilarityBasedClusterer(sim_rep, "single").cluster(self.database.get_tables())
        sim_al = SimilarityBasedClusterer(sim_rep, "average").cluster(self.database.get_tables())
        sim_cl = SimilarityBasedClusterer(sim_rep, "complete").cluster(self.database.get_tables())
        sim_cluster = meta_clusterer.cluster([sim_sl, sim_al, sim_cl])
        graph_spc = LinkClust(graph_rep, "betweenness").cluster(self.database.get_tables())
        graph_sp = LinkClust(graph_rep, "spectral_graph").cluster(self.database.get_tables())
        graph_cluster = meta_clusterer.cluster([graph_spc, graph_sp])

        clusters = meta_clusterer.cluster([vec_cluster, sim_cluster, graph_cluster])
        return clusters, time.time()-start_time


        
