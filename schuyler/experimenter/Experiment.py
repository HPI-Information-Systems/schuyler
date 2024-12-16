import wandb
import json
from io import BytesIO
import os

from schuyler.metrics.calculate_metrics import calculate_metrics

class Experiment:
    def __init__(self, name, database_name, database, scenario_id, group, base_scenario, scenario, solution, sql_file_path, meta_file_path, groundtruth_mapping_path, tag, use_wandb=False):
        run = wandb.init(
            project="schuyler",
            mode = "online" if use_wandb else "disabled",
            tags=[tag],
    		entity="Lasklu",
            config={
                "database_name": database_name,
                "group": group,
                "system": solution.solution_name,
                "scenario": scenario,
                "base_scenario": base_scenario,
                "sql_file_path": sql_file_path,
                "groundtruth_mapping_path": groundtruth_mapping_path,
                "meta_file_path": meta_file_path
            })
        self.name = name
        self.database = database
        self.scenario_id = scenario_id
        self.database_name = database_name
        self.sql_file_path = sql_file_path
        self.groundtruth_mapping_path = groundtruth_mapping_path
        with open(meta_file_path) as json_file:
                self.meta = json.load(json_file)
        self.solution = solution

    def setup(self, requires_pks=False):
        print("Setting up experiment")
        file_ending_is_json =  self.groundtruth_mapping_path.endswith(".json")
        print("Loading groundtruth mapping")
        if file_ending_is_json:
            with open(self.groundtruth_mapping_path) as json_file:
                data = json.load(json_file)
            self.groundtruth_mapping = JsonMapping(data, self.database_name, self.meta).to_D2RQ_Mapping()
            
            print("JSONGroundtruth mapping loaded")
        elif self.groundtruth_mapping_path.endswith(".ttl"):
            self.groundtruth_mapping = D2RQMapping(self.groundtruth_mapping_path, self.database_name, self.meta)
        elif os.path.isdir(self.groundtruth_mapping_path):
            merged_data = {}
            for file in os.listdir(self.groundtruth_mapping_path):
                if file.endswith(".json") and file != "meta.json":
                    path = os.path.join(self.groundtruth_mapping_path, file)
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for key, value in data.items():
                            if key in merged_data:
                                if isinstance(value, list) and isinstance(merged_data[key], list):
                                    merged_data[key].extend(value)
                                elif isinstance(value, dict) and isinstance(merged_data[key], dict):
                                    merged_data[key].update(value)
                                else:
                                    merged_data[key] = value
                            else:
                                merged_data[key] = value
            self.groundtruth_mapping = JsonMapping(merged_data, self.database_name, self.meta).to_D2RQ_Mapping()
            #calculate distinct attributes by having attributes that have the same name and belong to same class
        else:
            raise ValueError("File ending not supported")
        # all_attributes = self.groundtruth_mapping.get_relations()
        # for relation in self.groundtruth_mapping.get_relations():
        #     relation.set_eq_strategy(distinct=True)
        # distinct_attributes = []
        # for attribute in all_attributes:
        #     if attribute not in distinct_attributes:
        #         # print(attribute)
        #         # print("\n")
        #         distinct_attributes.append(attribute)
        # # print(all_attributes)
        # #print(distinct_attributes)
        # print("AMOUNT OF all RELATIONS", len(all_attributes))
        # print("AMOUNT OF DISTINCT RELATIONS", len(distinct_attributes))
        # raise Exception("STOP")

        with open(f"./output/groundtruths/{self.scenario_id}.ttl", "w") as f:
            f.write(self.groundtruth_mapping.create_ttl_string(self.database_name))
        wandb.save(f"./output/groundtruths/{self.scenario_id}.ttl")
        print("Rewriting database")
        print("Requires PKs", requires_pks)
        self.database.update_database(self.sql_file_path, add_primary_keys=requires_pks)        
        print("Experiment setup complete")
    
    def run(self, solution_config):
        requires_pks = solution_config["requires_pks"] if "requires_pks" in solution_config else False
        self.setup(requires_pks=requires_pks)
        train_config = solution_config["train"]
        test_config = solution_config["test"]
        trained_model, training_time = self.solution.train(**train_config)
        print(test_config)
        output_mapping, inference_time = self.solution.test(**test_config, model=trained_model, meta=self.meta, database_name=self.database_name)
        d2rq = output_mapping.create_ttl_string(self.database_name)
        output_path = os.path.join("./output", self.solution.solution_name, f"{self.scenario_id}.ttl")
        print(output_path)
        with open(output_path, "w") as f:
            f.write(d2rq)
        wandb.save("output.ttl")
        metrics = self.evaluate(self.groundtruth_mapping, output_mapping)    
        return {"output_mapping": output_mapping, "metrics": metrics, "runtime": {"training": training_time, "inference": inference_time}}  
    
    def evaluate(self, groundtruth_mapping: D2RQMapping, output_mapping: D2RQMapping):
        metrics = calculate_metrics(groundtruth_mapping, output_mapping)
        wandb.log(metrics)
        return metrics
    
    def save_to_file(self, ttl, filepath):
        file_like_object = BytesIO()
        graph = rdflib.Graph()
        graph.parse(data=ttl, format="turtle")
        graph.serialize(file_like_object, format='turtle')
        with open(filepath, "wb") as f:
            f.write(file_like_object.getvalue())

    def save_ttl_to_wandb(self, filename, ttl):
        file_like_object = BytesIO()
        graph = rdflib.Graph()
        graph.parse(data=ttl, format="turtle")
        graph.serialize(file_like_object, format='turtle')
        with open(filename, "wb") as f:
            f.write(file_like_object.getvalue())
        wandb.save(filename)
        # artifact = wandb.Artifact("graph.ttl", type='dataset')
        # artifact.add_file("./graph.ttl", file_like_object)