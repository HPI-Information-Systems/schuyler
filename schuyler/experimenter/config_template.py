from schuyler.solutions.iDisc.preprocessor import VectorRepresentator, SimilarityBasedRepresentator

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
                "meta_clusterer": {
                            "module": None,
                            "params": {
                                "None": "None"
                            }
                },
                "representators": {
                    "vector": 
                        [{
                            "module": VectorRepresentator,
                            "params": {
                            }
                        }],
                    "similarity": 
                        [
                            {
                                "module": SimilarityBasedRepresentator,
                                "params":{}
                            }
                        ],
                    "graph": 
                        []
                }
            }
        },
}

single_scenario = {
    "nm_tables": {
        
    }
}

scenarios = {
    "tpc_e": {
        "database_name": "real_world__tpce__orginal",
        "sql_file": "./data/tpc_e/script.sql",
        "groundtruth_file": "./data/tpc_e/groundtruth.csv",
    }
    
    }

experiment_config = {
    "scenarios": scenarios,
    "rewrite_database": False,
    "systems": [
        {
            "name": "iDisc",
            "config": systems["iDisc"]
        },
    ]
}

dynamic_config = lambda scenario, system: { "scenarios": scenario, "systems": [ { "name": system, "config": systems[system] } ] }

experiment_configs = {
    "base_experiment": experiment_config,
    "single_scenario": dynamic_config(single_scenario, "iDisc")
}