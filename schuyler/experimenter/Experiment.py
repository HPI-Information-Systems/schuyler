import wandb
import json
from io import BytesIO
import os

from schuyler.metrics.calculate_metrics import calculate_metrics

class Experiment:
    , tag=self.tag, use_wandb=self.use_wandb
    def __init__(self, name, database_name, solution, database, sql_file_path, groundtruth_path, tag, use_wandb=False):
        run = wandb.init(
            project="schuyler",
            mode = "online" if use_wandb else "disabled",
            tags=[tag],
    		entity="Lasklu",
            config={
                "database_name": database_name,
                "system": solution.solution_name,
                "sql_file_path": sql_file_path,
                "groundtruth_path": groundtruth_path,
            })
        self.name = name
        self.database = database
        self.database_name = database_name
        self.sql_file_path = sql_file_path
        self.solution = solution

    def setup(self):
        print("Setting up experiment")
        print("Loading groundtruth")
        #todo Load groundtruth
        self.groundtruth = None
        print("Rewriting database")
        self.database.update_database(self.sql_file_path)        
        print("Experiment setup complete")
    
    def run(self, solution_config):
        self.setup()
        train_config = solution_config["train"]
        test_config = solution_config["test"]
        trained_model, training_time = self.solution.train(**train_config)
        print(test_config)
        output_mapping, inference_time = self.solution.test(**test_config, model=trained_model, database=self.database)
        wandb.save("output.ttl")
        metrics = self.evaluate(self.groundtruth_mapping, output_mapping)    
        return {"output_mapping": output_mapping, "metrics": metrics, "runtime": {"training": training_time, "inference": inference_time}}  
    
    def evaluate(self, groundtruth, output):
        metrics = calculate_metrics(groundtruth, output)
        wandb.log(metrics)
        return metrics