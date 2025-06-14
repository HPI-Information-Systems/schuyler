import wandb
import os

from schuyler.metrics.calculate_metrics import calculate_metrics
from schuyler.experimenter.result import Result

class Experiment:
    def __init__(self, name, database_name, solution, database, sql_file_path, schema_file_path, groundtruth_path,hierarchy_level, tag, use_wandb=False, seed=42):
        
        # os.environ["WANDB_DIR"] = "/tmp"
        wandb.init(
            project="schuyler",
            mode = "online" if use_wandb else "disabled",
            tags=[tag],
    		entity="Lasklu",
            dir="/tmp/models",
            config={
                "database_name": database_name,
                "system": solution.solution_name,
                "sql_file_path": sql_file_path,
                "schema_file_path": schema_file_path,
                "groundtruth_path": groundtruth_path,
                "hierarchy_level": hierarchy_level,
                "seed": seed,
            })
        self.name = name
        self.database = database
        self.database_name = database_name
        self.sql_file_path = sql_file_path
        self.schema_file_path = schema_file_path
        self.hierarchy_level = hierarchy_level
        self.groundtruth_path = groundtruth_path
        self.solution = solution
        self.seed = seed

    def setup(self):
        print("Setting up experiment")
        print("Loading groundtruth")
        #todo Load groundtruth, it is a yaml file
        self.groundtruth = Result(hierarchy_level= self.hierarchy_level, path=self.groundtruth_path)
        print("Experiment setup complete")
    
    def run(self, solution_config):
        self.setup()
        print("Seed", self.seed)
        train_config = solution_config["train"]
        test_config = solution_config["test"]
        trained_model, training_time = self.solution.train(**train_config)
        wandb.log({f"test_{k}": v for k, v in test_config.items()})
        output, inference_time = self.solution.test(**test_config, sql_file_path=self.sql_file_path, schema_file_path=self.schema_file_path, groundtruth=self.groundtruth,model=trained_model, seed=self.seed)
        output = Result(hierarchy_level=self.hierarchy_level, data=output)
        wandb.log({"cluster_result": output.clusters})
        # output = self.groundtruth
        print("Output:", output)
        metrics = self.evaluate(output, self.groundtruth)    
        return { "metrics": metrics, "runtime": {"training": training_time, "inference": inference_time}}  
    
    def evaluate(self, output, groundtruth):
        metrics = calculate_metrics(output, groundtruth)
        print("Metrics:", metrics)
        wandb.log(metrics)
        return metrics