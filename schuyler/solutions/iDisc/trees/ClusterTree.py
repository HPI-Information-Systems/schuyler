import networkx as nx
from schuyler.solutions.iDisc.meta_clusterer.meta_clusterer import MetaClusterer
from schuyler.solutions.iDisc.trees.BaseClusterTree import BaseClusterTree

class ClusterTree(BaseClusterTree):
    def __init__(self, database):
        super().__init__(database)

    def construct_tree(self, base_clusterers):
        self.tree = nx.DiGraph()
        meta_clusterer = MetaClusterer(self.database)
        self.tree.add_node(meta_clusterer)
        for clusterer in base_clusterers:
            self.tree.add_node(clusterer)
            self.tree.add_edge(meta_clusterer, clusterer)
        return self.tree