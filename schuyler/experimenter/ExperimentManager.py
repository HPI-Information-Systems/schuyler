import wandb
import importlib
import os

from schuyler.experimenter.Experiment import Experiment
from schuyler.experimenter.config_template import experiment_configs
from schuyler.database.database import Database

class ExperimentManager():
    def __init__(self, tag, use_wandb=False) -> None:
        self.experiment_config = experiment_configs
        self.use_wandb = use_wandb
        self.tag = tag

    def start_experiments(self, experiment_name):
        print(experiment_name)
        self.experiment_config = self.experiment_config[experiment_name]
        for scenario, scenario_config in self.experiment_config["scenarios"].items():
            print("Running scenario: ", scenario)
            database_name = scenario_config["database_name"]
            schema = scenario_config.get("schema", None)
            if self.experiment_config["rewrite_database"]:
                print("Rewriting database")
                Database.update_database(scenario_config["sql_file"])
            else:
                print("Not rewriting database as defined in config.")   
            database_conn = Database(
                username=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                database=database_name,
                schema=schema
                )
            for system in self.experiment_config["systems"]:
                system_config = system["config"]
                system_name = system["name"]
                print(f"Running experiment {experiment_name} for database {database_name} with system {system_name}")
                metric_result, runtime = self.run_experiment(experiment_name=experiment_name, database_name=database_name, system_name=system_name, system_config=system_config,db_con=database_conn, sql_file_path=scenario_config["sql_file"], schema_file_path=scenario_config["schema_file"], groundtruth_path=scenario_config["groundtruth_file"], hierarchy_level=scenario_config.get("hierarchy_level", 0))
                print("Results:", metric_result)
                # wandb.log(metric_result)
                # wandb.log({"training_time": runtime["training"], "inference_time": runtime["inference"]})
                
        
    def run_experiment(self, experiment_name, database_name, system_name, system_config, sql_file_path, schema_file_path, groundtruth_path, db_con, hierarchy_level):
        if system_name == "iDisc":
            module = importlib.import_module("schuyler.solutions.iDisc.iDisc")
            system = getattr(module, "iDiscSolution")
        elif system_name == "schuyler":
            module = importlib.import_module("schuyler.solutions.schuyler.schuyler")
            system = getattr(module, "SchuylerSolution")
        elif system_name == "kCluster":
            module = importlib.import_module("schuyler.solutions.kCluster.kCluster")
            system = getattr(module, "kClusterSolution")
        elif system_name == "gpt":
            module = importlib.import_module("schuyler.solutions.gpt")
            system = getattr(module, "GPTSolution")
        elif system_name == "comdet":
            module = importlib.import_module("schuyler.solutions.comdet.comdet")
            system = getattr(module, "ComDetSolution")
        elif system_name == "comdet_clustering":
            module = importlib.import_module("schuyler.solutions.comdet_clustering")
            system = getattr(module, "ComDetClusteringSolution")
        elif system_name == "clustering":
            module = importlib.import_module("schuyler.solutions.clustering")
            system = getattr(module, "ClusteringSolution")
        elif system_name == "node2vec":
            module = importlib.import_module("schuyler.solutions.node2vec")
            system = getattr(module, "Node2VecSolution")
        else:
            raise ValueError("System not found")
        system = system(db_con)
        for i in range(1):
            seed = i + 42
            seed = 43
            print(f"Running experiment {experiment_name} for database {database_name} with system {system_name} and seed {seed}")
            experiment = Experiment(experiment_name, database_name=database_name, solution=system, database=db_con, sql_file_path=sql_file_path, schema_file_path=schema_file_path, groundtruth_path=groundtruth_path, hierarchy_level=hierarchy_level, tag=self.tag, use_wandb=self.use_wandb, seed=seed)
            try:
                output = experiment.run(solution_config=system_config)
                print("Output:", output["metrics"])
                wandb.finish()
            except Exception:
                print("An error occurred during the experiment!")
                import traceback
                e = traceback.format_exc()
                print(e)
                wandb.log({"error": str(e)})
                wandb.finish(exit_code=1)
                return None, None
        return output["metrics"], output["runtime"]

    