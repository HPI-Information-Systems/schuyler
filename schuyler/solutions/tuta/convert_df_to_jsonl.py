import json
import pandas as pd

def dataframe_to_wikitables_json(df: pd.DataFrame, table_id=1, title="") -> str:
    print("Dataframe to wikitables json")
    print(df)
    result = {
        "Title": title,
        "ID": table_id,
        "MergedRegions": [],
    }
    
    texts = []
    texts.append([str(col) for col in df.columns])
    for _, row_data in df.iterrows():
        texts.append([str(val)[:20] for val in row_data.values])
    
    result["Texts"] = texts
    
    result["TopHeaderRowsNumber"] = 1
    result["Height"] = len(texts)
    result["Width"] = len(texts[0]) if texts else 0
    

    result["LeftHeaderColumnsNumber"] = 1
    

    left_tree_root = {
        "CI": -1,
        "Cd": [],
        "RI": -1
    }
    for row_idx in range(result["Height"]):
        left_tree_root["Cd"].append({
            "CI": 0,
            "Cd": [],
            "RI": row_idx
        })
    result["LeftTreeRoot"] = left_tree_root
    
    top_tree_root = {
        "CI": -1,
        "Cd": [],
        "RI": -1
    }
    for col_idx in range(result["Width"]):
        top_tree_root["Cd"].append({
            "CI": col_idx,
            "Cd": [],
            "RI": 0
        })
    result["TopTreeRoot"] = top_tree_root
    print(result)
    return result

def write_to_jsonl(tables, i, output_file: str, title=""):
    print(tables)
    with open(output_file, "w") as f:
        for table in tables:
            print("Writing to jsonl")
            json.dump(dataframe_to_wikitables_json(table.get_df(10), i, table.table_name), f)
            f.write("\n")#
    return output_file
