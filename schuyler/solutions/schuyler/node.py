from schuyler.database.table import Table
from sentence_transformers import util
import numpy as np
class Node:
    def __init__(self, table: Table, llm_description, model):
        self.table = table
        self.llm_description = llm_description
        table.llm_description = self.llm_description
        self.encoding = np.asarray(model.encode(table), dtype="object")
        self.model = model
        
    def update_encoding(self, model):
        self.model = model
        self.model.model.cuda()
        self.encoding = self.model.encode(self.table)
    
    def calculate_table_similarity(self, node):
        return util.cos_sim(self.encoding.astype(np.float32), node.encoding.astype(np.float32)).item()

    def average_semantic_similarity_to_other_nodes(self, nodes):
        similarities = []
        for node in nodes:
            similarities.append(self.calculate_table_similarity(node))
        return sum(similarities) / len(similarities)
    
    def calculate_feature_vector(self, g):
        amount_of_fks = len(self.table.get_foreign_keys())
        amount_of_columns = len(self.table.columns)
        row_count = self.table.get_row_count()
        average_semantic_similarity = self.average_semantic_similarity_to_other_nodes(g.nodes)
        return {
            "embeddings": self.encoding,
            "amount_of_fks": amount_of_fks,
            "amount_of_columns": amount_of_columns,
            "row_count": row_count,
            "average_semantic_similarity": average_semantic_similarity
        }
    
    def is_reference_table(self, threshold=0.50):
        fk_columns = [col for fk in self.table.get_foreign_keys() for col in fk["constrained_columns"]]
        columns = [col["name"] for col in self.table.columns]
        if len(columns) > 1 and len(fk_columns) / len(columns) >= threshold and len(fk_columns) > 1:
            return True
        return False
    
    def __str__(self):
        return self.table.table_name
    
    def __repr__(self):
        return self.table.table_name