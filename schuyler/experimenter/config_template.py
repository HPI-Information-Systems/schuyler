from schuyler.solutions.iDisc.preprocessor import VectorRepresentator, SimilarityBasedRepresentator, GraphRepresentator
from schuyler.solutions.iDisc.trees import HierarchicalClusterTree, ClusterTree
from schuyler.solutions.iDisc.preprocessor.document_builder import TableNameDocumentBuilder, AttributeValuesDocumentBuilder, TableNameAndColsDocumentBuilder

systems = {
        "iDisc": {
            "train": {
            },
            "test": {
                "sim_clusterers": [{
                    "linkage": "average",
                    "metric": "cosine",
                    "representator": {
                        "module": SimilarityBasedRepresentator,
                        "params": {}
                    }
                },
                {
                    "linkage": "average",
                    "metric": "cosine",
                    "representator": {
                        "module": VectorRepresentator,
                        "params": {
                            "document_builder": TableNameAndColsDocumentBuilder
                        }
                    }
                },
                {
                    "linkage": "average",
                    "metric": "cosine",
                    "representator": {
                        "module": VectorRepresentator,
                        "params": {
                            "document_builder": TableNameDocumentBuilder
                        }
                    }
                },

                
                ],
                "link_clusterers": [{
                    "del_method": "spectral_graph",
                    "representator": {
                        "module": GraphRepresentator,
                        "params": {}
                    }
                }],
                # "meta_clusterer": {
                #     "module": None,
                #     "params": {
                #         "None": "None"
                #     }
                # },
                "tree": ClusterTree
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