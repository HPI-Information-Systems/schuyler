systems = {
        
}

single_scenario = {
    "nm_tables": {
        "trinary_relation": {
                "student_instructor_1": {
                    #"database_name": "hierarchy__two_tables__reviewer_1",
                    "sql_file": "train_data/nm_tables/trinary_relation/student_instructor_1/schema.sql",
                    "groundtruth_mapping": "train_data/nm_tables/trinary_relation/student_instructor_1/mapping.json",
                    "meta_file_path": "./evaluator/mapping_parser/d2rq_mapping/base_meta.json"
                },
        }
    }
}

scenarios = {
    "real_world": {
        "rba": {
            "original": {
                "sql_file": "./real-world/rba/create.sql",
            },
        },
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