import pandas as pd

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
        ]
        means = df[metric_cols].mean()
        stds = df[metric_cols].std()

        for col in metric_cols:
            df[col + "_zscore"] = (df[col] - means[col]) / (stds[col] if stds[col] != 0 else 1.0)
        zscore_cols = [col + "_zscore" for col in metric_cols]
        df["sum_of_zscores"] = df[zscore_cols].sum(axis=1)
        df_sorted = df.sort_values("sum_of_zscores", ascending=False)
        return df_sorted

    def get_reference_table_groups(self):
        equal_fk_groups = {k: v for k, v in self.foreign_key_set_of_reference_tables().items() if len(v) > 1}
        res = []
        for group in equal_fk_groups.values():
            for node in group:
                referred_tables = [fk["referred_table"] for fk in node.table.get_foreign_keys()]
                most_sim_node = self.G.get_node(max(referred_tables, key=lambda x: self.sim_matrix.loc[node.table.table_name, x]))
                res.append((most_sim_node, node))
        return res
        
    def get_entity_tables(self, attribute="sum_of_zscores"):
        table_ranking = self.rank_tables()
        table_ranking["node"] = table_ranking["table_name"].apply(lambda x: self.G.get_node(x))
        threshold_value = table_ranking[attribute].mean() + table_ranking[attribute].std()
        threshold = table_ranking[table_ranking[attribute] > threshold_value].shape[0]
        table_ranking = table_ranking[table_ranking["node"].apply(lambda x: not x.is_reference_table())]
        table_ranking[attribute] = (table_ranking[attribute] - table_ranking[attribute].min()) / (table_ranking[attribute].max() - table_ranking[attribute].min())
        table_ranking.to_csv("/data/database_schema_analysis.csv")
        entities = []
        i = 0
        table_ranking = table_ranking[table_ranking["degree"] > 0]
        while table_ranking.shape[0] > 0:
            row = table_ranking.iloc[0]
            node = row["node"]
            entities.append(node)
            neighbours = list(self.G.graph.neighbors(node))

            for neighbour in neighbours:
                table_ranking = table_ranking[table_ranking["table_name"] != neighbour.table.table_name]

            table_ranking = table_ranking[table_ranking["table_name"] != node.table.table_name]
            table_ranking = table_ranking.sort_values(attribute, ascending=False)
            i += 1
        return entities
             

    def foreign_key_set_of_reference_tables(self):
        fks = {}
        reference_tables = list(filter(lambda x: x.is_reference_table(), self.G.graph.nodes))
        for node in reference_tables:
            fk_columns = []
            foreign_keys = node.table.get_foreign_keys()
            if not foreign_keys:
                continue
            for fk in foreign_keys:
                fk_meta_data = [fk["referred_table"], sorted(fk["constrained_columns"])]
                fk_columns.append(fk_meta_data)
            fk_hash = hash(str(fk_columns))
            if fk_hash not in fks:
                fks[fk_hash] = []
            fks[fk_hash].append(node)
        return fks
