from schuyler.database.table import Table
from schuyler.solutions.schuyler.feature_vector.llm import LLM
from sentence_transformers import util
from schuyler.solutions.schuyler.feature_vector.representative_records import get_representative_records
import numpy as np

from string import Template
import os
import pandas as pd

class Node:
    def __init__(self, table: Table, llm, model, prompt_base_path, description_type, groundtruth_label=None):
        self.table = table
        self.llm = llm
        self.description_type = description_type
        self.llm_description = self.create_table_description(table, llm, prompt_base_path)
        table.llm_description = self.llm_description
        #self.encoding = np.asarray(model.encode(self.llm_description).cpu(), dtype="object")#node.encoding.cpu(),  dtype="object")
        self.encoding = np.asarray(model.encode(table), dtype="object")#node.encoding.cpu(),  dtype="object")
        self.model = model
        self.groundtruth_label = groundtruth_label
        #print(self.llm_description)

    def create_table_description(self, table: Table, llm: LLM, prompt_base_path, gt_label=None):#="/experiment/schuyler/solutions/schuyler/description.prompt", gt_label=None) -> str:
        database_name = table.db.database.split("__")[1]
        if llm.__name__ == "ChatGPT":
            folder = f"{self.description_type}_gpt"
        else:
            folder = self.description_type
        result_folder = f"/data/{database_name}/results/{folder}"#/{table.table_name}.txt"
        result_file = f"/data/{database_name}/results/{folder}/{table.table_name}.txt"
        os.makedirs(result_folder, exist_ok=True)
    
        #do only llm prediction if resulf_file does not exist
        if os.path.exists(result_file):
            print("Result file exists")
            with open(result_file, "r") as f:
                return f.read()
        else:
            representation = self.build_table_text_representation(table)
            prompt = Template(open(os.path.join(prompt_base_path, f"{self.description_type}.prompt")).read()).substitute(representation)
            print("Result file does not exist", result_file)
            if llm.__name__ == "ChatGPT":
                #print("Prompt", prompt)
                pred = llm.predict(prompt)
            else:
                pred = llm.predict(prompt).split("Description:")[-1].strip()
            #os.makedirs(f"/data/{database_name}/results/{self.description_type}/", exist_ok=True)
            with open(result_file, "w") as f:
                f.write(pred)
            return pred
        
    def update_encoding(self, model):
        self.model = model
        self.model.model.cuda()
        #self.encoding = np.asarray(self.model.encode(self.table).cpu(), dtype="object")
        self.encoding = self.model.encode(self.table)
    
    def build_table_text_representation(self, table: Table):
        table_name = table.table_name
        columns = [col["name"] for col in table.columns]
        data_samples = get_representative_records(table, 5)
        if data_samples is None:
           data_samples = table.get_df(5)

        # data_samples = table.get_df(5)
        
        #data_samples = table.get_df(5)#pd.DataFrame({col["name"]: table._get_data(col["name"], 5) for col in table.columns})
        fk_description = ""
        for fk in table.get_foreign_keys():
            fk_description += f" Foreign key '{' '.join(fk['constrained_columns'])}' references '{fk['referred_table']}'."
        #get foreign keys pointing to that table
        #fk_description += " Foreign keys pointing to this table: "

        tables = self.table.db.get_tables()
        for t in tables:
            for fk in t.get_foreign_keys():
                if fk["referred_table"] == table_name:
                    fk_description += f" Table '{t.table_name}' has a foreign key '{' '.join(fk['constrained_columns'])}' pointing to this table."
        
        database_context = "This table is part of the following database schema: "
        for t in tables:
            if t.table_name == table_name:
                continue
            database_context += f"Table '{t.table_name}' "
        
        primary_key = ", ".join(table.get_primary_key())
        return {
            "table_name": table_name,
            "schema": columns,
            "sample_data": data_samples,
            "fk_description": fk_description,
            #"database_context": database_context,
            "pk": primary_key
        }
    
    def calculate_table_similarity(self, node):
        return util.cos_sim(self.encoding.astype(np.float32), node.encoding.astype(np.float32)).item()

    def average_semantic_similarity_to_other_nodes(self, nodes):
        similarities = []
        for node in nodes:
            similarities.append(self.calculate_table_similarity(node))
        return sum(similarities) / len(similarities)
    
    # node features
    def calculate_feature_vector(self, g):
        print("Calculating features")
        amount_of_fks = len(self.table.get_foreign_keys())
        print("Amount of fks", amount_of_fks)
        amount_of_columns = len(self.table.columns)
        print("Amount of columns", amount_of_columns)
        row_count = self.table.get_row_count()
        print("Row count", row_count)
        average_semantic_similarity = self.average_semantic_similarity_to_other_nodes(g.nodes)
        print("Average semantic similarity", average_semantic_similarity)
        # return [amount_of_fks, amount_of_columns, row_count, average_semantic_similarity]
        return {
            "embeddings": self.encoding,
            "amount_of_fks": amount_of_fks,
            "amount_of_columns": amount_of_columns,
            "row_count": row_count,
            "average_semantic_similarity": average_semantic_similarity
        }
    
    def is_reference_table(self, threshold=0.49):
        fk_columns = [col for fk in self.table.get_foreign_keys() for col in fk["constrained_columns"]]

        #print("FK columns", fk_columns, "fpr table", self.table.table_name)
        columns = [col["name"] for col in self.table.columns]
        pks = self.table.get_primary_key()
        # drop fks from pks

        # pks = list(set(pks) - set(fk_columns))
        columns = list(set(columns) - set(pks))

        #no_of_columns_but_no_fk = len(set(columns) - set(fk_columns))
        if len(columns) > 0 and len(fk_columns) / len(columns) > threshold:
            return True
        return False
    
    def __str__(self):
        return self.table.table_name
    
    def __repr__(self):
        return self.table.table_name