from abc import ABC, abstractmethod
import networkx as nx

class BaseClusterTree(ABC):
    def __init__(self, database):
        self.database = database
    
    def cluster(self):
        clustering_results = {}
        for node in nx.topological_sort(self.tree.reverse()):
            if self.tree.out_degree(node) == 0:
                clustering_results[node] = node.cluster(self.database.get_tables())
            else: 
                clusterings = [clustering_results[child] for child in self.tree.successors(node)]
                clustering_results[node] = node.cluster(clusterings)
        root_node = [n for n,d in self.tree.in_degree() if d==0][0]
        print("Final clustering results", clustering_results[root_node])
        return clustering_results[root_node]        
    
    @abstractmethod
    def construct_tree(self, base_clusterers):
        raise NotImplementedError