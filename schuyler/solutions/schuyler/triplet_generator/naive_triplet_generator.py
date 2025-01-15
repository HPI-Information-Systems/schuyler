from schuyler.solutions.schuyler.triplet_generator.triplet_generator import BaseTripletGenerator
import networkx as nx
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sentence_transformers import util
import wandb
from schuyler.solutions.schuyler.node import Node


class NaiveTripletGenerator(BaseTripletGenerator):
    def __init__(self, database, G, sim_matrix, groundtruth):
        super().__init__(database, groundtruth)
        self.G = G
        self.sim_matrix = sim_matrix

    def enrich_triplets(self, triplets):
        anchors = [anchor.llm_description for anchor, _, _ in triplets]
        positives = [positive.llm_description for _, positive, _ in triplets]
        negatives = [negative.llm_description for _, _, negative in triplets]
        data = {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        }

        return Dataset.from_dict(data)

    def generate_triplets(self):
        triplets = []
        for node in self.G.graph.nodes:
            anchor = node
            # select most similar other node of all other nodes
            positive = max(self.G.graph.nodes, key=lambda n: self.sim_matrix.loc[str(anchor), str(n)])
            #random negative 
            negative = np.random.choice(list(self.G.graph.nodes))
            triplets.append((anchor, positive, negative))
        self.analyze_triplet_selection(triplets)
        # return self.enrich_pairs(triplets)
        return self.enrich_triplets(triplets)
    
    def convert_to_pairs(self, triplets):
        pairs = []
        for anchor, positive, negative in triplets:
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
        return pairs

    
    def identify_entity_tables(self, entity):
        entity_tables = []
        for node in self.G.nodes:
            if entity in node:
                entity_tables.append(node)
        return entity_tables
