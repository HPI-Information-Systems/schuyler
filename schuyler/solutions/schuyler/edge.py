from sentence_transformers import util
from schuyler.solutions.schuyler.node import Node
import numpy as np

class Edge:
    def __init__(self, node1: Node, node2: Node, st, sim=None):
        self.node1 = node1
        self.node2 = node2
        self.st = st
        self.weight = None
        if sim == None:
            self.table_sim = self.get_table_similarity()
        else:
            self.table_sim = sim

    def normalize(self, vector):
        if vector.is_cuda:
            vector = vector.cpu()
        vector_np = vector.numpy()
        return vector_np / np.linalg.norm(vector_np)

    def set_weight_attr_to_attr(self, attr):
        self.__setattr__("weight", self.__getattribute__(attr))

    def get_table_similarity(self):
        return util.cos_sim(self.node1.encoding.astype(float), self.node2.encoding.astype(float)).item()