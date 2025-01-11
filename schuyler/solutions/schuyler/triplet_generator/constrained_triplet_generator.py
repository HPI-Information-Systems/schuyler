from schuyler.solutions.schuyler.triplet_generator.triplet_generator import BaseTripletGenerator
import networkx as nx
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sentence_transformers import util
import wandb
from schuyler.solutions.schuyler.node import Node


class ConstrainedTripletGenerator(BaseTripletGenerator):
    def __init__(self, database, G, sim_matrix, groundtruth):
        super().__init__(database, groundtruth)
        self.G = G
        self.sim_matrix = sim_matrix
        self.schema_analyzer = DatabaseSchemaAnalyzer(self.database, self.G, self.sim_matrix)

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

    def enrich_pairs(self, triplets):
        anchors = [anchor.llm_description for anchor, _, _ in triplets]
        positives = [positive.llm_description for _, positive, _ in triplets]
        negatives = [negative.llm_description for _, _, negative in triplets]
        labels = [1] * len(anchors) + [0] * len(anchors)
        data = {
            "sentence1": anchors + anchors,
            "sentence2": positives + negatives,
            "label": labels
        }
        
        dataset = Dataset.from_dict(data)
        return dataset.shuffle(seed=42)
    

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
            #negative = self.G.get_node(max(entity_tables, key=lambda x: self.sim_matrix.loc[anchor.table.table_name, x.table.table_name]))
            triplets.append((positive, anchor, negative))
        for entity_table in entity_tables:
            print("Entity table", entity_table)
            neighbors = list(self.G.graph.neighbors(entity_table))
            neighbors = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, neighbors))
            positive = self.G.get_node(max(neighbors, key=lambda x: self.sim_matrix.loc[entity_table.table.table_name, x.table.table_name]))
            neighbors = list(filter(lambda x: x.table.table_name != positive.table.table_name, neighbors))
            if not neighbors:
                # randomly select a negative
                negative = self.G.get_node(np.random.choice(self.G.graph.nodes))
            else:
                print("Neighbors", neighbors)
                negative = self.G.get_node(min(neighbors, key=lambda x: self.sim_matrix.loc[positive.table.table_name, x.table.table_name]))
                # negative = self.G.get_node(np.random.choice(self.G.graph.nodes))
            triplets.append((entity_table, positive, negative))
            # randomly select four more negatives that do not share any edges
            negative_candidates = list(filter(lambda x: x not in neighbors, entity_tables))
            negative_candidates = list(filter(lambda x: x.table.table_name != entity_table.table.table_name, negative_candidates))
            i = 0
            while i < 3 and negative_candidates:
                print("Negative candidates", negative_candidates)
                negative = self.G.get_node(max(negative_candidates, key=lambda x: self.sim_matrix.loc[positive.table.table_name, x.table.table_name]))
                # negative = self.G.get_node(np.random.choice(negative_candidates))
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
    

class DatabaseSchemaAnalyzer():
    def __init__(self, database, G, sim_matrix):
        self.database = database
        self.G = G
        self.sim_matrix = sim_matrix
    
    def rank_tables(self):
        page_rank = []
        degree = []
        betweeness = []
        average_semantic_similarity = []
        amount_of_fks = []
        amount_of_columns = []
        row_count = []

        df = []
        for node in self.G.graph.nodes:
            df.append({"table_name": node.table.table_name, "page_rank": node.page_rank, "degree": node.degree, "betweeness": node.betweenness_centrality, "average_semantic_similarity": node.average_semantic_similarity, "amount_of_fks": node.amount_of_fks, "amount_of_columns": node.amount_of_columns, "row_count": node.row_count})
            page_rank.append(node.page_rank)
            degree.append(node.degree)
            betweeness.append(node.betweenness_centrality)
            average_semantic_similarity.append(node.average_semantic_similarity)
            amount_of_fks.append(node.amount_of_fks)
            amount_of_columns.append(node.amount_of_columns)
            row_count.append(node.row_count)
        df = pd.DataFrame(df)
        metric_cols = [
        "page_rank",
        "degree",
        "betweeness",
        #"average_semantic_similarity",
        # "amount_of_fks",
        # "amount_of_columns",
        #"row_count"
        ]
        means = df[metric_cols].mean()
        stds = df[metric_cols].std()

        for col in metric_cols:
            df[col + "_zscore"] = (df[col] - means[col]) / (stds[col] if stds[col] != 0 else 1.0)
        zscore_cols = [col + "_zscore" for col in metric_cols]
        df["sum_of_zscores"] = df[zscore_cols].sum(axis=1)
        # df_positive_deviation = df[(df[zscore_cols] >= 0).all(axis=1)]
        df_sorted = df.sort_values("sum_of_zscores", ascending=False)
        # df_sorted["sum_of_zscores"] = (df_sorted["sum_of_zscores"] - df_sorted["sum_of_zscores"].min()) / (df_sorted["sum_of_zscores"].max() - df_sorted["sum_of_zscores"].min())

        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df_sorted.head(20))
        return df_sorted

    def get_reference_table_groups(self):
        equal_fk_groups = {k: v for k, v in self.foreign_key_set_of_reference_tables().items() if len(v) > 1}
        # print("Equal fk groups", equal_fk_groups)   
        res = []
        for group in equal_fk_groups.values():
            for node in group:
                referred_tables = [fk["referred_table"] for fk in node.table.get_foreign_keys()]
                # print("Referred tables", referred_tables)
                most_sim_node = self.G.get_node(max(referred_tables, key=lambda x: self.sim_matrix.loc[node.table.table_name, x]))
                res.append((most_sim_node, node))
        return res
        
    def get_entity_tables(self, attribute="sum_of_zscores"):
        table_ranking = self.rank_tables()
        table_ranking["node"] = table_ranking["table_name"].apply(lambda x: self.G.get_node(x))
        threshold_value = table_ranking[attribute].mean() + table_ranking[attribute].std()
        threshold = table_ranking[table_ranking[attribute] > threshold_value].shape[0]
        print(len(table_ranking))
        table_ranking = table_ranking[table_ranking["node"].apply(lambda x: not x.is_reference_table())]
        table_ranking[attribute] = (table_ranking[attribute] - table_ranking[attribute].min()) / (table_ranking[attribute].max() - table_ranking[attribute].min())
        table_ranking.to_csv("/data/database_schema_analysis.csv")
        print(len(table_ranking))
        entities = []
        #minmax of sum_of_zscores
        print(table_ranking[attribute])
        #while table_ranking.iloc[0]["sum_of_zscores"] - table_ranking.iloc[1]["sum_of_zscores"] < 0.3:
        i = 0
        print("Threshold", threshold, "Threshold value", threshold_value)
        #drop all tables without edges 
        table_ranking = table_ranking[table_ranking["degree"] > 0]
        while table_ranking.shape[0] > 0:
            row = table_ranking.iloc[0]
            node = row["node"]
            entities.append(node)
            neighbours = list(self.G.graph.neighbors(node))
            print("Drop neighbours", neighbours)
            #drop all neighbours from table ranking
            for neighbour in neighbours:
                table_ranking = table_ranking[table_ranking["table_name"] != neighbour.table.table_name]

            # for neighbour in neighbours:
            #     table_ranking.loc[table_ranking["table_name"] == neighbour.table.table_name, attribute] -= 0.02
            table_ranking = table_ranking[table_ranking["table_name"] != node.table.table_name]
            table_ranking = table_ranking.sort_values(attribute, ascending=False)
            print("Table ranking", table_ranking)
            i += 1
        print("Entities", entities)
        return entities
             

    def foreign_key_set_of_reference_tables(self):
        fks = {}
        reference_tables = list(filter(lambda x: x.is_reference_table(), self.G.graph.nodes))
        # print(reference_tables)
        for node in reference_tables:
            fk_columns = []
            foreign_keys = node.table.get_foreign_keys()
            if not foreign_keys:
                continue
            # print("Foreign keys", foreign_keys, "for table", node.table.table_name)
            for fk in foreign_keys:
                fk_meta_data = [fk["referred_table"], sorted(fk["constrained_columns"])]
                # print("fk_meta_data", fk_meta_data)
                fk_columns.append(fk_meta_data)
            fk_hash = hash(str(fk_columns))
            if fk_hash not in fks:
                fks[fk_hash] = []
            fks[fk_hash].append(node)
        return fks
