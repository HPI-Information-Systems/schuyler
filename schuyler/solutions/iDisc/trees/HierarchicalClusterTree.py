from collections import defaultdict
import networkx as nx
import numpy as np

from schuyler.solutions.iDisc.base_clusterer import SimilarityBasedClusterer, LinkClust
from schuyler.solutions.iDisc.meta_clusterer.meta_clusterer import MetaClusterer
def is_not_empty(x): return x if len(x) > 0 else None

class HierarchicalClusterTree:
    def __init__(self, database):
        super().__init__(database)

    def construct_tree(self, base_clusterers):
        w = base_clusterers
        self.tree = nx.DiGraph()
        while len(w) > 1:
            max_sim_clusterers = self.determine_max_similarity_level_of_base_clusterers(w)[0]
            print("max_sim_clusterers", max_sim_clusterers)
            print("w", w)
            meta_clusterer = MetaClusterer(self.database)
            self.tree.add_node(meta_clusterer)
            print(max_sim_clusterers)
            for clusterer in max_sim_clusterers:
                print(clusterer)
                self.tree.add_node(clusterer)
                self.tree.add_edge(meta_clusterer, clusterer)
                w.remove(clusterer)
            w.append(meta_clusterer)
        self.tree.add_node(w[0])
        for node in self.tree.nodes:
            if self.tree.in_degree(node) == 0 and node != w[0]:
                self.tree.add_edge(w[0], node)
        return self.tree
    
    def determine_max_similarity_level_of_base_clusterers(self, clusterers):
        base_clusterers = list(filter(self.is_base_clusterer, clusterers))
        meta_clusterers = list(filter(self.is_meta_clusterer, clusterers))
        first_level = self.first_similarity_level(base_clusterers)
        second_level = self.second_similarity_level(base_clusterers)# if any(self.second_similarity_level(clusterers)) else None
        third_level = base_clusterers# self.third_similarity_level(clusterers) if any(self.third_similarity_level(clusterers)) else None
        third_level.extend(meta_clusterers)
        #first_level.append(meta_clusterers) if not second_level else second_level.append(meta_clusterers) if not third_level else third_level.append(meta_clusterers)
        return first_level if first_level else second_level if second_level else [third_level]

    def get_nested_attr(instance, attr_path):
        attrs = attr_path.split(".")
        for attr in attrs:
            instance = getattr(instance, attr)
        return instance

    def first_similarity_level(self, clusterers):
        vector_groups = self.group_by(self.is_representator(clusterers, "VectorRepresentator"), 'representator.document_builder.name', 'representator.vectorizer')
        similarity_groups = self.group_by(self.is_representator(clusterers, "SimilarityBasedRepresentator"), 'linkage', 'metric', 'representator.metric', 'representator.amount_of_values_to_load')
        graph_groups = self.group_by(self.is_representator(clusterers, "GraphRepresentator"))
        res = []
        res.extend(vector_groups) if is_not_empty(vector_groups) else None
        res.extend(similarity_groups) if is_not_empty(similarity_groups) else None
        res.extend(graph_groups) if is_not_empty(graph_groups) else None
        print(res)
        return res if len(res) > 0 else None
        return (is_not_empty(vector_groups), is_not_empty(similarity_groups), is_not_empty(graph_groups))
        #return is_not_empty(similarity_representator), is_not_empty(vector_representator), is_not_empty(graph_representator)
    
    def second_similarity_level(self, clusterers):
        similarity_representator = is_not_empty(self.is_representator(clusterers, "SimilarityBasedRepresentator"))
        vector_representator = is_not_empty(self.is_representator(clusterers, "VectorRepresentator"))
        graph_representator = is_not_empty(self.is_representator(clusterers, "GraphRepresentator"))
        res = []
        res.append(similarity_representator) if similarity_representator else None
        res.append(vector_representator) if vector_representator else None
        res.append(graph_representator) if graph_representator else None
        #remove all with length of 1
        res = list(filter(lambda x: len(x) > 1, res))
        return res if len(res) > 0 else None
        return is_not_empty(similarity_representator), is_not_empty(vector_representator), is_not_empty(graph_representator)

    def third_similarity_level(self, clusterers):
        #basically if the other two similarity levels do not hold anything
        return clusterers

    def is_base_clusterer(self, clusterer): return isinstance(clusterer, SimilarityBasedClusterer) or isinstance(clusterer, LinkClust)

    def is_meta_clusterer(self, clusterer): return not self.is_base_clusterer(clusterer)

    def is_representator(self, clusterers, representator_name): 
        return list(filter(lambda x: x.representator.name == representator_name, clusterers)) 
    
    def group_by(self, instances, *attributes):
        grouped = defaultdict(list)
        for instance in instances:
            key = tuple(get_nested_attr(instance, attr) for attr in attributes)
            grouped[key].append(instance)
        grouped = {key: value for key, value in grouped.items() if len(value) > 1}
        return list(grouped.values())
    
def get_nested_attr(instance, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        instance = getattr(instance, attr)
    return instance