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
        self.temp_clustering = []

    def generate_triplets(self):
        triplets = []
        entity_tables = self.schema_analyzer.get_entity_tables()
        print(entity_tables)
        # print("Entity tables", entity_tables)
        # # get entity detail tables -> have one-to-one relationship with an entity table
        for entity_table in entity_tables:
            candidate_neighbors = [
                neighbor for neighbor in self.G.graph.neighbors(entity_table)
                if neighbor not in entity_tables
            ]

            valid_neighbors = []
            for neighbor in candidate_neighbors:
                foreign_keys = neighbor.table.get_foreign_keys()
                if foreign_keys:
                    all_fk_match = all(
                        fk["referred_table"] == entity_table.table.table_name
                        for fk in foreign_keys
                    )
                    pk = neighbor.table.get_primary_key()
                    def fk_part_of_pk(fk, pk):
                        return all(col in pk for col in fk["referred_columns"])
                    pk_fk_match = all(
                        fk_part_of_pk(fk, pk)
                        for fk in foreign_keys
                    )
                    # pk_fk_match = all(
                    #     fk["referred_columns"] == pk
                    #     for fk in foreign_keys
                    # )
                    if all_fk_match and pk_fk_match and not neighbor.is_reference_table():
                        valid_neighbors.append(neighbor)
                else:
                    continue

            print("Entity table", entity_table, "Valid neighbors", valid_neighbors)
            for neighbor in valid_neighbors:
                self.add_tables_to_clustering(entity_table.table.table_name, neighbor.table.table_name)
                random_table = self.G.get_node(np.random.choice(entity_tables))
                triplet = (entity_table, neighbor, random_table)
                triplets.append(triplet)
        print("DETECTED triplets", triplets)

        #     # get all tables that have a foreign key to this table
        #     foreign_key_tables = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, self.G.graph.neighbors(entity_table)))
        #     reference_tables = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, self.G.graph.predecessors(entity_table)))



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
            self.add_tables_to_clustering(anchor.table.table_name, positive.table.table_name)
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
            self.add_tables_to_clustering(entity_table.table.table_name, positive.table.table_name)
            # randomly select four more negatives that do not share any edges
            # negative = self.G.get_node(np.random.choice(negative_candidates))
            # triplets.append((entity_table, positive, negative))
            i = 0
            used_candidates = {}
            candidates = list(self.G.graph.nodes())
            while i < 5:
                neighbors = list(self.G.graph.neighbors(entity_table))
                negative_candidates = list(filter(lambda x: x not in neighbors, entity_tables))
                negative_candidates = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, negative_candidates))
                average_similarity = self.sim_matrix[entity_table.table.table_name].quantile(0.25)
                if entity_table.table.table_name not in used_candidates:
                    used_candidates[entity_table.table.table_name] = []
                valid = [
                    x
                    for x in candidates
                    if x.table.table_name not in used_candidates[entity_table.table.table_name] and self.sim_matrix.loc[entity_table.table.table_name, x.table.table_name] > average_similarity
                ]
                if not valid:
                    continue
                negative = self.G.get_node(min(valid, key=lambda x: self.sim_matrix.loc[entity_table.table.table_name, x.table.table_name]))
                used_candidates[entity_table.table.table_name].append(negative.table.table_name)
                # negative_candidates.remove(negative)
                triplets.append((entity_table, positive, negative))
                i += 1
            # entity_tables.remove(positive) if positive in entity_tables else None
            # triplets.append((entity_table, positive, negative))
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

    def are_tables_in_same_cluster(self, table1, table2):
        for cluster in self.temp_clustering:
            if table1 in cluster and table2 in cluster:
                return True
        return False
        
    def add_tables_to_clustering(self, table1: str, table2: str) -> None:
        idx1 = idx2 = None

        # Find which clusters (if any) contain table1 and table2
        for idx, cluster in enumerate(self.temp_clustering):
            if table1 in cluster:
                idx1 = idx
            if table2 in cluster:
                idx2 = idx

        # Case A: both in the same cluster → nothing to do
        if idx1 is not None and idx1 == idx2:
            return

        # Case B: both in different clusters → merge them
        if idx1 is not None and idx2 is not None and idx1 != idx2:
            keep, remove = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
            # merge, avoiding duplicates
            merged = self.temp_clustering[keep] + [
                t for t in self.temp_clustering[remove]
                if t not in self.temp_clustering[keep]
            ]
            self.temp_clustering[keep] = merged
            # drop the other cluster
            del self.temp_clustering[remove]
            return

        # Case C: exactly one exists → add the other
        if idx1 is not None:
            if table2 not in self.temp_clustering[idx1]:
                self.temp_clustering[idx1].append(table2)
            return
        if idx2 is not None:
            if table1 not in self.temp_clustering[idx2]:
                self.temp_clustering[idx2].append(table1)
            return

        # Case D: neither exists → new cluster
        self.temp_clustering.append([table1, table2])