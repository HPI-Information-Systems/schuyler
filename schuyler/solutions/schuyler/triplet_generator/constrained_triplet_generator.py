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
        anchor_positive_pairs = {
            "detail": [],
            "reference": [],
            "entity": []
        }
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
                    if all_fk_match and pk_fk_match and not neighbor.is_reference_table():
                        valid_neighbors.append(neighbor)
                else:
                    continue

            print("Entity table", entity_table, "Valid neighbors", valid_neighbors)
            for neighbor in valid_neighbors:
                self.add_tables_to_clustering(entity_table.table.table_name, neighbor.table.table_name)
                random_table = self.G.get_node(np.random.choice(entity_tables))
                # while self.are_tables_in_same_cluster(entity_table.table.table_name, random.table.table_name):
                #     random_table = self.G.get_node(np.random.choice(entity_tables))
                anchor_positive_pairs["detail"].append([entity_table, neighbor])
                # triplet = (entity_table, neighbor, random_table)
                # triplets.append(triplet)


        reference_table_groups = self.schema_analyzer.get_reference_table_groups()
        reference_table_dict = {}
        for anchor, positive in reference_table_groups:
            if anchor not in reference_table_dict:
                reference_table_dict[anchor] = []
            reference_table_dict[anchor].append(positive)


        for reference_table_group in reference_table_groups:
            anchor = reference_table_group[0]
            positive = reference_table_group[1]
            self.add_tables_to_clustering(anchor.table.table_name, positive.table.table_name)
            entity_tables = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != positive.table.table_name, entity_tables))
            candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name, self.G.graph.nodes))
            # negative = self.G.get_node(np.random.choice(candidates))
            # while self.are_tables_in_same_cluster(anchor.table.table_name, negative.table.table_name):
            #     negative = self.G.get_node(np.random.choice(candidates))
            # triplets.append((positive, anchor, negative))
            anchor_positive_pairs["reference"].append([positive, anchor])
        
        for entity_table in entity_tables:
            neighbors = list(self.G.graph.neighbors(entity_table))
            neighbors = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, neighbors))
            if not neighbors:
                continue
            positive = self.G.get_node(max(neighbors, key=lambda x: self.sim_matrix.loc[entity_table.table.table_name, x.table.table_name]))

            neighbors = list(filter(lambda x: x.table.table_name != positive.table.table_name, neighbors))
            self.add_tables_to_clustering(entity_table.table.table_name, positive.table.table_name)
            negative_candidates = list(filter(lambda x: x not in neighbors, entity_tables))
            negative_candidates = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, negative_candidates))
            negative = self.G.get_node(np.random.choice(negative_candidates))
            while self.are_tables_in_same_cluster(entity_table.table.table_name, negative.table.table_name):
                negative = self.G.get_node(np.random.choice(negative_candidates))
            anchor_positive_pairs["entity"].append([entity_table, positive])
            # triplets.append((entity_table, positive, negative))
            i = 0
            while i < 4 and negative_candidates:
                self.add_tables_to_clustering(entity_table.table.table_name, positive.table.table_name)
                negative = self.G.get_node(np.random.choice(negative_candidates))
                while self.are_tables_in_same_cluster(entity_table.table.table_name, negative.table.table_name):
                    negative = self.G.get_node(np.random.choice(negative_candidates))
                negative_candidates.remove(negative)
                anchor_positive_pairs["entity"].append([entity_table, positive])
                # triplets.append((entity_table, positive, negative))
                i += 1
            entity_tables.remove(positive) if positive in entity_tables else None
            # triplets.append((entity_table, positive, negative))
        # negative for detail tables
        # candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name, self.G.graph.nodes))
        candidates = self.schema_analyzer.get_entity_tables()
        candidates = list(self.G.graph.nodes())
        used_candidates = {}
        for pair in anchor_positive_pairs["detail"]:
            # for i in range(3):
            random_table = self.G.get_node(np.random.choice(list(self.G.graph.nodes)))
            if pair[0].table.table_name not in used_candidates:
                used_candidates[pair[0].table.table_name] = []
            negative = self.select_negative_record(pair[0], pair[1], candidates, used_candidates, 0.3)
            used_candidates[pair[0].table.table_name].append(negative.table.table_name)
            triplets.append((pair[0], pair[1], negative))
        # used_candidates = {}
        for pair in anchor_positive_pairs["reference"]:
            # candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != pair[0].table.table_name, entity_tables))
            # candidates = list(filter(lambda x: x.table.table_name != pair[0].table.table_name, self.G.graph.nodes))
            if pair[0].table.table_name not in used_candidates:
                used_candidates[pair[0].table.table_name] = []
            random_table = self.G.get_node(np.random.choice(list(self.G.graph.nodes)))
            # while self.are_tables_in_same_cluster(pair[0].table.table_name, random_table.table.table_name):
            #     random_table = self.G.get_node(np.random.choice(entity_tables))
            negative = self.select_negative_record(pair[0], pair[1], candidates, used_candidates, 0.3)
            used_candidates[pair[0].table.table_name].append(negative.table.table_name)
            triplets.append((pair[0], pair[1], negative))
        # used_candidates = {}
        # candidates = self.schema_analyzer.get_entity_tables()
        for pair in anchor_positive_pairs["entity"]:
            #!
            # neighbors = list(self.G.graph.neighbors(pair[0]))
            # negative_candidates = list(filter(lambda x: x not in neighbors, entity_tables))
            # negative_candidates = list(filter(lambda x: x.table.table_name != pair[0].table.table_name, negative_candidates))
            # average_similarity = self.sim_matrix[pair[0].table.table_name].quantile(0.25)
            # # candidates = 
            if pair[0].table.table_name not in used_candidates:
                used_candidates[pair[0].table.table_name] = []
            # valid = [
            #     x
            #     for x in candidates
            #     if x.table.table_name not in used_candidates[pair[0].table.table_name] and self.sim_matrix.loc[pair[0].table.table_name, x.table.table_name] > average_similarity
            # ]
            # if not valid:
            #     continue
            # negative = self.G.get_node(min(valid, key=lambda x: self.sim_matrix.loc[pair[0].table.table_name, x.table.table_name]))
            #!
            # negative = self.G.get_node(np.random.choice(list(self.G.graph.nodes)))
            # negative = self.G.get_node(np.random.choice(negative_candidates))
            # used_candidates[pair[0].table.table_name].append(negative.table.table_name)
            # candidates.remove(negative)
            # while self.are_tables_in_same_cluster(pair[0].table.table_name, random_table.table.table_name) and self.are_tables_in_same_cluster(pair[1].table.table_name, random_table.table.table_name):
            #     random_table = self.G.get_node(np.random.choice(candidates))
            negative = self.select_negative_record(pair[0], pair[1], candidates, used_candidates, 0.3)
            used_candidates[pair[0].table.table_name].append(negative.table.table_name)
            triplets.append((pair[0], pair[1], negative))
            # if pair[0] in entity_tables:
            #     entity_tables.remove(pair[0])
        print("Generated triplets")
        print(self.temp_clustering)
        print(anchor_positive_pairs)
        print(triplets)
        self.analyze_triplet_selection(triplets)
        return triplets
    
    def convert_to_pairs(self, triplets):
        pairs = []
        for anchor, positive, negative in triplets:
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
        return pairs

    def select_negative_record(self, anchor, positive, candidates, already_used, margin):
        anchor_positive_sim = self.sim_matrix.loc[anchor.table.table_name, positive.table.table_name]
        negative = None
        # find all candidates those similarity is less than anchor_positive_sim - margin
        candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != positive.table.table_name, candidates))
        candidates = list(filter(lambda x: x.table.table_name not in already_used, candidates))
        els = list(filter(lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name] < anchor_positive_sim - margin, candidates))
        valid = [
                x
                for x in els
                if x.table.table_name not in already_used[anchor.table.table_name]
            ]
    
        if valid:
            # select the one with the minimum similarity
            negative = self.G.get_node(max(valid, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))
            #select the one closest to the mean similarity of thise
            # average_similarity = self.sim_matrix.loc[anchor.table.table_name, [x.table.table_name for x in els]].mean()
            # #closest to the mean
            # negative = self.G.get_node(min(els, key=lambda x: abs(self.sim_matrix.loc[anchor.table.table_name, x.table.table_name] - average_similarity)))
            # negative = self.G.get_node(np.random.choice(candidates))

        if negative is None:
            average_similarity = self.sim_matrix[anchor.table.table_name].quantile(0.25)
            valid = [
                x
                for x in candidates
                if x.table.table_name not in already_used and self.sim_matrix.loc[anchor.table.table_name, x.table.table_name] > average_similarity
            ]
            if not valid:
                return None
            negative = self.G.get_node(min(valid, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))
        return negative

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
        for idx, cluster in enumerate(self.temp_clustering):
            if table1 in cluster:
                idx1 = idx
            if table2 in cluster:
                idx2 = idx

        if idx1 is not None and idx1 == idx2:
            return
        if idx1 is not None and idx2 is not None and idx1 != idx2:
            keep, remove = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
            merged = self.temp_clustering[keep] + [
                t for t in self.temp_clustering[remove]
                if t not in self.temp_clustering[keep]
            ]
            self.temp_clustering[keep] = merged
            del self.temp_clustering[remove]
            return

        if idx1 is not None:
            if table2 not in self.temp_clustering[idx1]:
                self.temp_clustering[idx1].append(table2)
            return
        if idx2 is not None:
            if table1 not in self.temp_clustering[idx2]:
                self.temp_clustering[idx2].append(table1)
            return

        self.temp_clustering.append([table1, table2])