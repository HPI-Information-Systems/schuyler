from schuyler.solutions.schuyler.triplet_generator.triplet_generator import BaseTripletGenerator
import numpy as np
from schuyler.solutions.schuyler.triplet_generator.database_schema_analyzer import DatabaseSchemaAnalyzer
class ConstrainedTripletGenerator(BaseTripletGenerator):
    def __init__(self, database, G, sim_matrix):
        super().__init__(database)
        self.G = G
        self.sim_matrix = sim_matrix
        self.schema_analyzer = DatabaseSchemaAnalyzer(self.database, self.G, self.sim_matrix)
        self.temp_clustering = []

    def generate_triplets(self):
        triplets = []
        entity_tables = list(self.schema_analyzer.get_entity_tables())  # list of node objects
        anchor_positive_pairs = {"detail": [], "reference": [], "entity": []}
        if not hasattr(self, "_seen_pairs"):
            self._seen_pairs = set()
        for entity_table in list(entity_tables):
            entity_name = entity_table.table.table_name

            candidate_neighbors = [
                n for n in self.G.graph.neighbors(entity_table)
                if n not in entity_tables
            ]

            valid_neighbors = []
            for neighbor in candidate_neighbors:
                foreign_keys = neighbor.table.get_foreign_keys()
                if not foreign_keys:
                    continue

                all_fk_match = all(
                    fk["referred_table"] == entity_name for fk in foreign_keys
                )
                pk = neighbor.table.get_primary_key()

                def fk_part_of_pk(fk, pk_cols):
                    return all(col in pk_cols for col in fk["referred_columns"])

                pk_fk_match = all(fk_part_of_pk(fk, pk) for fk in foreign_keys)

                if all_fk_match and pk_fk_match and not neighbor.is_reference_table():
                    valid_neighbors.append(neighbor)

            name_to_neighbor = {n.table.table_name: n for n in valid_neighbors}
            neighbor_names = sorted(name_to_neighbor.keys())
            for nbr_name in neighbor_names:
                pair_key = tuple(sorted((entity_name, nbr_name)))
                if pair_key in self._seen_pairs:
                    continue
                self._seen_pairs.add(pair_key)
                neighbor_obj = name_to_neighbor[nbr_name]
                anchor_positive_pairs["detail"].append([entity_table, neighbor_obj])
        reference_table_groups = self.schema_analyzer.get_reference_table_groups()
        reference_table_dict = {}
        for anchor, positive in reference_table_groups:
            reference_table_dict.setdefault(anchor, []).append(positive)

        for anchor, positive in reference_table_groups:
            a_name = anchor.table.table_name
            p_name = positive.table.table_name

            _ = self.add_tables_to_clustering(a_name, p_name)
            entity_tables = [
                x for x in entity_tables
                if x.table.table_name not in (a_name, p_name)
            ]

            anchor_positive_pairs["reference"].append([positive, anchor])
        for entity_table in list(entity_tables):
            et_name = entity_table.table.table_name

            neighbors = [x for x in self.G.graph.neighbors(entity_table)
                        if x.table.table_name != et_name]
            if not neighbors:
                continue
            positive = self.G.get_node(
                max(neighbors, key=lambda x: self.sim_matrix.loc[et_name, x.table.table_name])
            )
            p_name = positive.table.table_name

            _ = self.add_tables_to_clustering(et_name, p_name)
            anchor_positive_pairs["entity"].append([entity_table, positive])

            i = 0
            negative_candidates = [x for x in entity_tables if (x not in neighbors and x.table.table_name != et_name)]
            while i < 4 and negative_candidates:
                _ = self.add_tables_to_clustering(et_name, p_name)
                neg_choice = np.random.choice(negative_candidates)
                if neg_choice in negative_candidates:
                    negative_candidates.remove(neg_choice)
                anchor_positive_pairs["entity"].append([entity_table, positive])
                i += 1

            if positive in entity_tables:
                entity_tables.remove(positive)

        candidates = list(self.G.graph.nodes())
        used_candidates = {}

        for a_node, p_node in anchor_positive_pairs["detail"]:
            a_name = a_node.table.table_name
            used_candidates.setdefault(a_name, [])
            negative = self.select_negative_record(a_node, p_node, candidates, used_candidates, 0.3)
            used_candidates[a_name].append(negative.table.table_name)
            triplets.append((a_node, p_node, negative))

        for p_node, a_node in anchor_positive_pairs["reference"]:
            a_name = p_node.table.table_name
            used_candidates.setdefault(a_name, [])
            negative = self.select_negative_record(p_node, a_node, candidates, used_candidates, 0.3)
            used_candidates[a_name].append(negative.table.table_name)
            triplets.append((p_node, a_node, negative))

        for a_node, p_node in anchor_positive_pairs["entity"]:
            a_name = a_node.table.table_name
            used_candidates.setdefault(a_name, [])
            negative = self.select_negative_record(a_node, p_node, candidates, used_candidates, 0.3)
            used_candidates[a_name].append(negative.table.table_name)
            triplets.append((a_node, p_node, negative))
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
        candidates = list(filter(lambda x: x.table.table_name != anchor.table.table_name and x.table.table_name != positive.table.table_name, candidates))
        candidates = list(filter(lambda x: x.table.table_name not in already_used, candidates))
        els = list(filter(lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name] < anchor_positive_sim - margin, candidates))
        valid = [
                x
                for x in els
                if x.table.table_name not in already_used[anchor.table.table_name]
            ]
    
        if valid:
            negative = self.G.get_node(max(valid, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))

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
    def add_tables_to_clustering(self, table1: str, table2: str) -> bool:
        idx1 = idx2 = None
        changed = False

        for idx, cluster in enumerate(self.temp_clustering):
            if table1 in cluster:
                idx1 = idx
            if table2 in cluster:
                idx2 = idx
            if idx1 is not None and idx2 is not None:
                break

        # already in same cluster → no-op
        if idx1 is not None and idx1 == idx2:
            return False

        # merge two different clusters
        if idx1 is not None and idx2 is not None:
            keep, remove = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
            for t in self.temp_clustering[remove]:
                if t not in self.temp_clustering[keep]:
                    self.temp_clustering[keep].append(t)
                    changed = True
            del self.temp_clustering[remove]
            return changed

        if idx1 is not None:
            if table2 not in self.temp_clustering[idx1]:
                self.temp_clustering[idx1].append(table2)
                changed = True
            return changed

        if idx2 is not None:
            if table1 not in self.temp_clustering[idx2]:
                self.temp_clustering[idx2].append(table1)
                changed = True
            return changed

        self.temp_clustering.append([table1, table2])
        return True    