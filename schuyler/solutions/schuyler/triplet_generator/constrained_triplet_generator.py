from schuyler.solutions.schuyler.triplet_generator.triplet_generator import BaseTripletGenerator
import networkx as nx
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sentence_transformers import util
import wandb
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.triplet_generator.database_schema_analyzer import DatabaseSchemaAnalyzer


class ConstrainedTripletGenerator(BaseTripletGenerator):
    def __init__(self, database, G, sim_matrix, groundtruth):
        super().__init__(database, groundtruth)
        self.G = G
        self.sim_matrix = sim_matrix
        self.schema_analyzer = DatabaseSchemaAnalyzer(self.database, self.G, self.sim_matrix)
        
    

    def generate_triplets(self):
        triplets = []
        entity_tables = self.schema_analyzer.get_entity_tables()
        print("Entity tables", entity_tables)
        reference_table_groups = self.schema_analyzer.get_reference_table_groups()
        print("Reference table groups", reference_table_groups)
        # group reference table group by anchor
        #reference_table_groups = {anchor: [positives for anchors, positives in reference_table_groups if anchors == anchor or positives == anchor] for anchor, positive in reference_table_groups}
        reference_table_dict = {}
        for anchor, positive in reference_table_groups:
            if anchor not in reference_table_dict:
                reference_table_dict[anchor] = []
            reference_table_dict[anchor].append(positive)
        print("Reference table dict", reference_table_dict)
        #print("Reference table groups", reference_table_groups)
        # for anchor, positives in reference_table_dict.items():
        #     added_positives = [anchor]
        #     for positive in positives:
        #         entity_tables = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != positive.table.table_name, entity_tables))
        #         #negative = self.G.get_node(max(entity_tables, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))
        #         added_positives.append(positive)
        #         #calculate average encoding of positives
        #         average_encoding = np.mean([positive.encoding.astype(float) for positive in added_positives], axis=0)
        #         sims = []
        #         for entity_table in entity_tables:
        #             cos_sim = util.cos_sim(average_encoding, entity_table.encoding.astype(float)).item()
        #             sims.append((entity_table, cos_sim))
        #         negative = max(sims, key=lambda x: x[1])[0]
        #         triplets.append((anchor, positive, negative))
        #         # add a negative that is neighbor of the positive and not the anchor
        #         # neighbors = list(self.G.graph.neighbors(positive))
        #         # neighbors = list(filter(lambda x: x.table.table_name != anchor.table.table_name, neighbors))
        #         # if not neighbors:
        #         #     negative = self.G.get_node(np.random.choice(self.G.graph.nodes))
        #         # else:
        #         #     negative = self.G.get_node(np.random.choice(neighbors))
        #         # triplets.append((positive, anchor, negative))
        #         #add a random negative from all nodes that is not in the positive set and not the anchor and not neighbor
        #         candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x not in added_positives, self.G.graph.nodes))
        #         negative = self.G.get_node(np.random.choice(candidates))
        #         triplets.append((positive, anchor, negative))
                
                #entity_tables.remove(positive)


        for reference_table_group in reference_table_groups:

            anchor = reference_table_group[0]
            positive = reference_table_group[1]
            entity_tables = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != positive.table.table_name, entity_tables))
            candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name, self.G.graph.nodes))
            negative = self.G.get_node(np.random.choice(candidates))
            print("Anchor", anchor, "Positive", positive, "Negative", negative)
            #negative = self.G.get_node(max(entity_tables, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))
            triplets.append((positive, anchor, negative))
        for entity_table in entity_tables:
            print("Entity table", entity_table)
            neighbors = list(self.G.graph.neighbors(entity_table))
            neighbors = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, neighbors))
            if not neighbors:
                continue
                #positive = self.G.get_node(np.random.choice(self.G.graph.nodes))
            positive = self.G.get_node(max(neighbors, key=lambda x: self.sim_matrix.loc[entity_table.table.table_name, x.table.table_name]))

            neighbors = list(filter(lambda x: x.table.table_name != positive.table.table_name, neighbors))
            if not neighbors:
                # randomly select a negative
                negative = self.G.get_node(np.random.choice(self.G.graph.nodes))
            else:
                print("Neighbors", neighbors)
                # negative = self.G.get_node(min(neighbors, key=lambda x: self.sim_matrix.loc[positive.table.table_name, x.table.table_name]))
                negative = self.G.get_node(np.random.choice(self.G.graph.nodes))
                print("Negative", negative)
            triplets.append((entity_table, positive, negative))
            # randomly select four more negatives that do not share any edges
            negative_candidates = list(filter(lambda x: x not in neighbors, entity_tables))
            negative_candidates = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, negative_candidates))
            i = 0
            while i < 3 and negative_candidates:
                print("Negative candidates", negative_candidates)
                # negative = self.G.get_node(min(negative_candidates, key=lambda x: self.sim_matrix.loc[positive.table.table_name, x.table.table_name]))
                # select median element
                # negative = negative_candidates[len(negative_candidates) // 2]
                #select mean element in negative candidates
                # negative = negative_candidates[np.argmin([self.sim_matrix.loc[positive.table.table_name, x.table_name] for x in negative_candidates])]

                print("Negative", negative)
                negative = self.G.get_node(np.random.choice(negative_candidates))
                # check 
                negative_candidates.remove(negative)
                triplets.append((entity_table, positive, negative))
                i += 1
            entity_tables.remove(positive) if positive in entity_tables else None
            triplets.append((entity_table, positive, negative))
        print("Generated triplets")
        print(triplets)
        self.analyze_triplet_selection(triplets)
        # return self.enrich_pairs(triplets)
        return triplets
    
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
    

