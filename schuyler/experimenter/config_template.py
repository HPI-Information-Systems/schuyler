from schuyler.solutions.schuyler.triplet_generator import ConstrainedTripletGenerator
from sklearn.cluster import AffinityPropagation
from schuyler.solutions.schuyler.feature_vector.llm import vLLM
systems = {
        "schuyler": {
            "train": {
            },
            "test": {
                "description_type": "description",
                "prompt_base_path": "/experiment/schuyler/solutions/schuyler/prompts/",
                "triplet_generation_model": ConstrainedTripletGenerator,
                "prompt_model": vLLM,
                "finetune": True,
                "clustering_method": AffinityPropagation,
            }
        },
        "node2vec": {
            "train": {
            },
            "test": {
                "prompt_base_path": "/experiment/schuyler/solutions/schuyler/prompts/",
                "description_type": "description_concise_stack_2_runtime_node2vec",
                "prompt_model": vLLM
            }
        },
        "gpt": {
            "train": {
            },
            "test": {
                "gpt_model": "gpt-4o"
            }
        },
        "comdet": {
            "train": {},
            "test": {}
        },
        "clustering": {
            "train": {},
            "test":{
                    "prompt_base_path": "/experiment/schuyler/solutions/schuyler/prompts/",
                    "prompt_model": vLLM,
                    "description_type": "description_concise_stack_2_runtime_cluster",
                    "clustering_method": AffinityPropagation,
                }
        },
        "comdet_clustering": {
            "train": {},
            "test": {
                "prompt_model": vLLM,
                  "description_type": "description_concise_stack_2_runtime_comclust",
                  "prompt_base_path": "/experiment/schuyler/solutions/schuyler/prompts/",
                }
        }
}

scenarios = {
    "tpc_e": {
        "database_name": "real_world__tpc_e__orginal",
        "sql_file": "/data/tpc_e/script.sql",
        "schema_file": "/data/tpc_e/script_only_schema.sql",
        "groundtruth_file": "/data/tpc_e/groundtruth.yaml",
    },
    "stack_exchange": {
        "database_name": "real_world__stack_exchange__original",
        "sql_file": "/data/stack_exchange/script.sql",
        "schema_file": "/data/stack_exchange/script_only_schema.sql",
        "groundtruth_file": "/data/stack_exchange/groundtruth.yaml"
    },
    "adventure_works": {
        "database_name": "real_world__adventure_works__original",
        "sql_file": "/data/adventure_works/backup_file.sql",
        "schema_file": "/data/adventure_works/backup_file_only_schema.sql",
        "groundtruth_file": "/data/adventure_works/groundtruth.yaml",
    },
    
    "magento": {
        "database_name": "real_world__magento__orginal",
        "sql_file": "/data/magento/script.sql",
        "schema_file": "/data/magento/script_only_schema.sql",
        "groundtruth_file": "/data/magento/groundtruth.yaml",
        "hierarchy_level": 1
    },
    "musicbrainz": {
        "database_name": "real_world__musicbrainz__original",
        "sql_file": "/data/musicbrainz/output_script.sql",
        "schema_file": "/data/musicbrainz/output_script_only_schema.sql",
        "groundtruth_file": "/data/musicbrainz/groundtruth.yaml",
    },
    }

experiment_config = {
    "scenarios": scenarios,
    "rewrite_database": False,
    "systems": [
        {
            "name": "schuyler",
            "config": systems["schuyler"]
        },
        # {
        #     "name": "node2vec",
        #     "config": systems["node2vec"]
        # },
        # {
        #     "name": "comdet",
        #     "config": systems["comdet"]
        # },
        # {
        #     "name": "comdet_clustering",
        #     "config": systems["comdet_clustering"]
        # },
        # {
        #     "name": "clustering",
        #     "config": systems["clustering"]
        # },
        # {
        #     "name": "gpt",
        #     "config": systems["gpt"]
        # }
    ]
}

dynamic_config = lambda scenario, system: { "scenarios": scenario, "systems": [ { "name": system, "config": systems[system] } ] }

experiment_configs = {
    "base_experiment": experiment_config,
}