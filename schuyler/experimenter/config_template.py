from schuyler.solutions.iDisc.preprocessor import VectorRepresentator

systems = {
        "iDisc": {
            "train": {
            },
            "test": {
                "sim_clust": {
                    "linkage": "average",
                    "metric": "cosine"
                },
                "link_clust": {
                    "linkage": "average",
                    "metric": "cosine"
                },
                "representators": {
                    "vector": 
                        [{
                            "module": VectorRepresentator,
                            "params": {
                                "None": "None"
                            }
                        }],
                    "similarity": 
                        [{
                            "module": None,
                            "params": {
                                "None": "None"
                            }
                        }],
                    "graph": 
                        [{
                            "module": None,
                            "params": {
                                "None": "None"
                            }
                        }]
                }
        },
}

single_scenario = {
    "nm_tables": {
        
    }
}

scenarios = {
    "tpc_e": {
        "database_name": "tpc_e",
        "sql_file": "./data/tpc_e/script.sql",
        "groundtruth_file": "./data/tpc_e/groundtruth.csv",
    }
    
    }

experiment_config = {
    "scenarios": scenarios,
    "systems": [
        {
            "name": "d2rmapper",
            "config": systems["d2rmapper"]
        },
    ]
}

dynamic_config = lambda scenario, system: { "scenarios": scenario, "systems": [ { "name": system, "config": systems[system] } ] }

experiment_configs = {
    "base_experiment": experiment_config,
    "single_scenario": dynamic_config(single_scenario, "d2rmapper")
}