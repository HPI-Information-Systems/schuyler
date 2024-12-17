import wandb
import importlib
import configparser
import rdflib
from io import BytesIO
import os

from schuyler.experimenter.Experiment import Experiment
from schuyler.database.database import Database

class ExperimentManager():
    def __init__(self, config_file, tag, use_wandb=False) -> None:
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.experiment_config = experiment_configs
        self.use_wandb = use_wandb
        self.tag = tag

    def start_experiments(self, experiment_name):
        print(experiment_name)
        self.experiment_config = self.experiment_config[experiment_name]
        for scenario, scenario_config in self.experiment_config["scenarios"].items():
            print("Running scenario: ", scenario)
            database_name = scenario_config["database_name"]
            database_conn = Database(
                username=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                database=database_name)
            for system in self.experiment_config["systems"]:
                system_config = system["config"]
                system_name = system["name"]
                print(f"Running experiment {experiment_name} for database {database_name} with system {system_name}")
                metric_result, runtime = self.run_experiment(experiment_name=experiment_name, database_name=database_name, system_name=system_name, system_config=system_config,db_con=database_conn, sql_file_path=scenario_config["sql_file"], groundtruth_path=scenario_config["groundtruth_file"])
                print("Results:", metric_result)
                wandb.log(metric_result)
                wandb.log({"training_time": runtime["training"], "inference_time": runtime["inference"]})
                wandb.finish()
        
    def run_experiment(self, experiment_name, database_name, system_name, system_config, sql_file_path, groundtruth_path, db_con):
        if system_name == "iDisc":
            module = importlib.import_module("schuyler.solutions.iDisc")
            system = getattr(module, "iDisc")
        else:
            raise ValueError("System not found")
        print("configuration", self.config)
        system = system(**self.config[system_name] if system_name in self.config else ValueError("System not found"))
        experiment = Experiment(experiment_name, database_name=database_name, solution=system, database=db_con, sql_file_path=sql_file_path, groundtruth_path=groundtruth_path, tag=self.tag, use_wandb=self.use_wandb)
        try:
            output = experiment.run(system_config)
        except Exception:
            print("An error occurred during the experiment!")
            import traceback
            e = traceback.format_exc()
            print(e)
            wandb.log({"error": str(e)})
            wandb.finish(exit_code=1)
            return None, None
        return output["metrics"], output["runtime"]

    