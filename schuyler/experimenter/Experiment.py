import wandb

from schuyler.metrics.calculate_metrics import calculate_metrics
from schuyler.experimenter.result import Result

class Experiment:
    def __init__(self, name, database_name, solution, database, sql_file_path, groundtruth_path, tag, use_wandb=False):
        wandb.init(
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
        self.groundtruth_path = groundtruth_path
        self.solution = solution

    def setup(self):
        print("Setting up experiment")
        print("Loading groundtruth")
        #todo Load groundtruth, it is a yaml file
        self.groundtruth = Result(path=self.groundtruth_path)
        print("Experiment setup complete")
    
    def run(self, solution_config):
        self.setup()
        train_config = solution_config["train"]
        test_config = solution_config["test"]
        trained_model, training_time = self.solution.train(**train_config)
        output, inference_time = self.solution.test(**test_config, model=trained_model)
        output = Result(data=output)
        print("Output:", output)
        metrics = self.evaluate(output, self.groundtruth)    
        return { "metrics": metrics, "runtime": {"training": training_time, "inference": inference_time}}  
    
    def evaluate(self, output, groundtruth):
        metrics = calculate_metrics(output, groundtruth)
        print("Metrics:", metrics)
        #wandb.log(metrics)
        return metrics