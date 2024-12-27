from schuyler.database.table import Table
from schuyler.solutions.schuyler.feature_vector.llm import LLM
from sentence_transformers import util

from string import Template
import os
import pandas as pd

class Node:
    def __init__(self, table: Table, llm, st):
        self.table = table
        self.llm = llm
        self.llm_description = self.create_table_description(table, llm)
        self.st = st
        print(self.llm_description)

    def create_table_description(self, table: Table, llm: LLM, path_to_prompt="/experiment/schuyler/solutions/schuyler/description.prompt") -> str:
        representation = self.build_table_text_representation(table)
        print(representation)
        prompt = Template(open(path_to_prompt).read()).substitute(representation)
        database_name = table.db.database.split("__")[1]
        result_file = f"/data/{database_name}/results/{table.table_name}.txt"
        #do only llm prediction if resulf_file does not exist
        if os.path.exists(result_file):
            print("Result file exists")
            with open(result_file, "r") as f:
                return f.read()
        else:
            print("Result file does not exist", result_file)
            pred = llm.predict(prompt).split("Description:")[-1].strip()
            os.makedirs(f"/data/{database_name}/results", exist_ok=True)
            with open(f"/data/{database_name}/results/{table.table_name}.txt", "w") as f:
                f.write(pred)
            return pred
    
    def build_table_text_representation(self, table: Table):
        table_name = table.table_name
        columns = [col["name"] for col in table.columns]
        data_samples = table.get_df(5)#pd.DataFrame({col["name"]: table._get_data(col["name"], 5) for col in table.columns})
        fk_description = ""
        for fk in table.get_foreign_keys():
            fk_description += f" Foreign key '{' '.join(fk['constrained_columns'])}' references '{fk['referred_table']}'."
        primary_key = ", ".join(table.get_primary_key())
        return {
            "table_name": table_name,
            "schema": columns,
            "sample_data": data_samples,
            "fk_description": fk_description,
            "pk": primary_key
        }
    
    def calculate_table_similarity(self, node):
        em1 = self.st.encode(self.llm_description)
        em2 = self.st.encode(node.llm_description)
        return util.cos_sim(em1, em2).item()

    def average_semantic_similarity_to_other_nodes(self, nodes):
        similarities = []
        for node in nodes:
            similarities.append(self.calculate_table_similarity(node))
        return sum(similarities) / len(similarities)
    
    # node features
    def calculate_feature_vector(self, g):
        amount_of_fks = len(self.table.get_foreign_keys())
        amount_of_columns = len(self.table.columns)
        row_count = self.table.get_row_count()
        average_semantic_similarity = self.average_semantic_similarity_to_other_nodes(g.nodes)
        embedding = self.st.encode(self.llm_description)
        # return [amount_of_fks, amount_of_columns, row_count, average_semantic_similarity]
        return {
            "embeddings": embedding,
            "amount_of_fks": amount_of_fks,
            "amount_of_columns": amount_of_columns,
            "row_count": row_count,
            "average_semantic_similarity": average_semantic_similarity
        }
    
    def __str__(self):
        return self.table.table_name
    
    def __repr__(self):
        return self.table.table_name