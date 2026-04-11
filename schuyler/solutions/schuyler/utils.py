import pandas as pd
from schuyler.solutions.schuyler.feature_vector.representative_records import parallel_representatives
import os
from string import Template
from schuyler.solutions.schuyler.feature_vector.llm import vLLM

def normalize_edge_weights(graph, weight_attribute="weight"):
    edge_weights = [graph[edge[0]][edge[1]].get(weight_attribute, 0) for edge in graph.edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    if max_weight == min_weight:
        raise ValueError("All edge weights are identical; normalization is not possible.")
    for edge in graph.edges:
        original_weight = graph[edge[0]][edge[1]].get(weight_attribute, 0)
        normalized_weight = (original_weight - min_weight) / (max_weight - min_weight)
        graph[edge[0]][edge[1]][weight_attribute] = normalized_weight
    return graph

def get_database_descriptions(database, prompt_base_path, description_type, iteration=0):
    prompts = {}
    folder = f"{description_type}_vLLM"
    result_folder = f"/data/{database.database.split('__')[1]}/results/{folder}/{iteration}" 
    prompt_folder = f"/data/{database.database.split('__')[1]}/results/{folder}/prompts" 
    os.makedirs(prompt_folder, exist_ok=True)
    prompt_file_missing = any([not os.path.exists(f"{prompt_folder}/{table.table_name}.prompt") for table in database.get_tables()])
    result_file_missing = any([not os.path.exists(f"{result_folder}/{table.table_name}.txt") for table in database.get_tables()])
    descriptions = {}
    if result_file_missing:
        llm = vLLM()
        if prompt_file_missing:
            tables = database.get_tables()
            foreign_keys = {}
            table_representation = parallel_representatives(database, k=5)
            primary_keys = {}
            for table in tables:
                foreign_keys[table.table_name] = table.get_foreign_keys()
                primary_keys[table.table_name] = table.get_primary_key()
            for table in tables:
                representation = table_representation.get(table.table_name, {})
                prompt = Template(open(os.path.join(prompt_base_path, f"{description_type}.prompt")).read()).substitute(representation)
                prompts[table.table_name] = prompt
                with open(f"{prompt_folder}/{table.table_name}.prompt", "w") as f:
                    f.write(prompt)
        else:
            print(f"Loading existing prompts from {prompt_folder}.")
            for table in database.get_tables():
                with open(f"{prompt_folder}/{table.table_name}.prompt", "r") as f:
                    prompts[table.table_name] = f.read()
        
        os.makedirs(result_folder, exist_ok=True)
        generated_descriptions = llm.predict(list(prompts.values()))
        descriptions = dict(zip(prompts.keys(), generated_descriptions))
        
        import gc, torch
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for table, description in descriptions.items():
            result_file = os.path.join(result_folder, f"{table}.txt")
            with open(result_file, "w") as f:
                    f.write(description)
    else:
        llm = None
        print(f"Loading existing descriptions from {result_folder}.")
        for table in database.get_tables():
            with open(f"{result_folder}/{table.table_name}.txt", "r") as f:
                descriptions[table.table_name] = f.read()
    return descriptions


