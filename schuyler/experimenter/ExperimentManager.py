import wandb
import importlib
import os
import copy

from schuyler.experimenter.Experiment import Experiment
from schuyler.experimenter.config_template import experiment_configs
from schuyler.database.database import Database
from schuyler.solutions.schuyler.feature_vector.llm import vLLM

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
            sql_file_path = scenario_config["sql_file"]
            if self.experiment_config["rewrite_database"]:
                print("Rewriting database")
                Database.update_database(sql_file_path)
            else:
                print("Not rewriting database as defined in config.")

            try:
                database_conn = Database(
                    username=os.getenv("POSTGRES_USER"),
                    password=os.getenv("POSTGRES_PASSWORD"),
                    host=os.getenv("POSTGRES_HOST"),
                    port=os.getenv("POSTGRES_PORT"),
                    database=database_name,
                    schema=schema
                )
            except ValueError as e:
                error_msg = str(e)
                if "does not exist" in error_msg and not self.experiment_config["rewrite_database"]:
                    print(f"Database '{database_name}' does not exist. Bootstrapping it from {sql_file_path}.")
                    Database.update_database(sql_file_path)
                    database_conn = Database(
                        username=os.getenv("POSTGRES_USER"),
                        password=os.getenv("POSTGRES_PASSWORD"),
                        host=os.getenv("POSTGRES_HOST"),
                        port=os.getenv("POSTGRES_PORT"),
                        database=database_name,
                        schema=schema
                    )
                else:
                    raise
            for system in self.experiment_config["systems"]:
                system_config = system["config"]
                system_name = system["name"]
                print(f"Running experiment {experiment_name} for database {database_name} with system {system_name}")
                metric_result, _ = self.run_experiment(experiment_name=experiment_name, database_name=database_name, system_name=system_name, system_config=system_config,db_con=database_conn, sql_file_path=sql_file_path, schema_file_path=scenario_config["schema_file"], groundtruth_path=scenario_config["groundtruth_file"], hierarchy_level=scenario_config.get("hierarchy_level", 0))
                print("Results:", metric_result)
                
        
    def run_experiment(self, experiment_name, database_name, system_name, system_config, sql_file_path, schema_file_path, groundtruth_path, db_con, hierarchy_level):
        if system_name == "schuyler":
            module = importlib.import_module("schuyler.solutions.schuyler.schuyler")
            system = getattr(module, "SchuylerSolution")
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
            seed = 43
            print(f"Running experiment {experiment_name} for database {database_name} with system {system_name} and seed {seed}")
            temp_sys_config = copy.deepcopy(system_config)
            print("System config:", system_config)
            if isinstance(system_config["test"].get("triplet_generation_model", None), list):
                print(system_config["test"]["triplet_generation_model"])
                for idx, model in enumerate(system_config["test"]["triplet_generation_model"]):
                    name = model.__name__
                    print(f"Running experiment {experiment_name} for database {database_name} with system {system_name} and seed {seed} and model {name}")
                    temp_sys_config["test"]["triplet_generation_model"] = model
                    metrics, runtime = self.do_experiment(experiment_name=f"{experiment_name}_{name}_{idx}", database_name=database_name, system=system, system_config=temp_sys_config, sql_file_path=sql_file_path, schema_file_path=schema_file_path, groundtruth_path=groundtruth_path, db_con=db_con, hierarchy_level=hierarchy_level, seed=seed, i=i)
            elif isinstance(system_config["test"].get("clustering_method", None), list):
                for clustering_method in system_config["test"]["clustering_method"]:
                    temp_sys_config["test"]["clustering_method"] = clustering_method
                    name = clustering_method.__name__
                    metrics, runtime = self.do_experiment(experiment_name=f"{experiment_name}_{name}", database_name=database_name, system=system, system_config=temp_sys_config, sql_file_path=sql_file_path, schema_file_path=schema_file_path, groundtruth_path=groundtruth_path, db_con=db_con, hierarchy_level=hierarchy_level, seed=seed, i=i)
            else:
                metrics, runtime = self.do_experiment(experiment_name=experiment_name, database_name=database_name, system=system, system_config=system_config, sql_file_path=sql_file_path, schema_file_path=schema_file_path, groundtruth_path=groundtruth_path, db_con=db_con, hierarchy_level=hierarchy_level, seed=seed, i=i)
        return metrics, runtime
    
    def do_experiment(self, experiment_name, database_name, system, system_config, sql_file_path, schema_file_path, groundtruth_path, db_con, hierarchy_level, seed, i):
        experiment = Experiment(experiment_name, database_name=database_name, solution=system, database=db_con, sql_file_path=sql_file_path, schema_file_path=schema_file_path, groundtruth_path=groundtruth_path, hierarchy_level=hierarchy_level, tag=self.tag, use_wandb=self.use_wandb, seed=seed, iteration=i)
        try:    
            output = experiment.run(solution_config=system_config)
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


    