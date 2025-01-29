from schuyler.solutions.schuyler.triplet_generator.triplet_generator import BaseTripletGenerator
import networkx as nx
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sentence_transformers import util
import wandb
from schuyler.solutions.schuyler.node import Node


class NeighborTripletGenerator(BaseTripletGenerator):
    def __init__(self, database, G, sim_matrix, groundtruth):
        super().__init__(database, groundtruth)
        self.G = G
        self.sim_matrix = sim_matrix

    def generate_triplets(self):
        triplets = []
        for node in self.G.graph.nodes:
            anchor = node
            #select random neighbor
            if list(self.G.graph.neighbors(anchor)):
                positive = np.random.choice(list(self.G.graph.neighbors(anchor)))
            else:
                positive = np.random.choice(list(self.G.graph.nodes))
            negative = np.random.choice(list(set(self.G.graph.nodes) - set(self.G.graph.neighbors(anchor))))

            #random negative 
            #select random non-neigbor
            #negative = np.random.choice(list(self.G.graph.nodes))
            triplets.append((anchor, positive, negative))
        self.analyze_triplet_selection(triplets)
        # return self.enrich_pairs(triplets)
        return triplets
