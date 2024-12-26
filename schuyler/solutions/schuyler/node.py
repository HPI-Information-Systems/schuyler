from schuyler.database.table import Table
from schuyler.solutions.schuyler.feature_vector.llm import LLM
from string import Template
import os
import pandas as pd

class Node:
    def __init__(self, table: Table, llm):
        self.table = table
        self.llm = llm
        self.llm_description = self.create_table_description(table, llm)
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
    
    def __str__(self):
        return self.table.table_name
    
    def __repr__(self):
        return self.table.table_name