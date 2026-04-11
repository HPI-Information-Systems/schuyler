from schuyler.database import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from multiprocessing import get_context

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def _init_worker():
    # vermeidet, dass jeder Prozess mit vielen Threads läuft
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
def _sanitize_cell(x):
    # Konvertiert problematische Zellen in pickelbare Strings
    if isinstance(x, memoryview):
        x = bytes(x)
    if isinstance(x, (bytes, bytearray)):
        # wähle eine Repräsentation: hex oder base64
        return "0x" + bytes(x).hex()
        # alternativ: return base64.b64encode(bytes(x)).decode("ascii")
    if isinstance(x, np.generic):
        return x.item()
    return x

def _sanitize_df_for_mp(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(_sanitize_cell)
def get_representative_records_df(record_values, columns, amount_of_records=5):
    if record_values is None or len(record_values) < amount_of_records:
        print("Not enough records to select representatives from.")
        return None

    documents = [" ".join(map(str, record)) for record in record_values]

    try:
        X = TfidfVectorizer().fit_transform(documents)
    except ValueError:
        return None

    kmeans = KMeans(n_clusters=amount_of_records, random_state=0, n_init="auto")
    kmeans.fit(X)

    representatives = []
    for cluster_idx in range(amount_of_records):
        cluster_indices = np.where(kmeans.labels_ == cluster_idx)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_vectors = X[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_idx]
        similarities = cosine_similarity(cluster_vectors, centroid.reshape(1, -1))
        closest_idx = cluster_indices[np.argmax(similarities)]
        representatives.append(record_values[closest_idx])

    if not representatives:
        return None
    return pd.DataFrame(representatives, columns=columns)

def parallel_representatives(database, k=5, max_workers=os.cpu_count()):
    print(f"Calculating representatives for database {database.database} using {max_workers} workers.")
    tables = database.get_tables()
    foreign_keys = {}
    primary_keys = {}
    columns = {}
    for table in tables:
        foreign_keys[table.table_name] = table.get_foreign_keys()
        primary_keys[table.table_name] = table.get_primary_key()
        columns[table.table_name] = [col["name"] for col in table.columns]
    total = len(tables)

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, mp_context=get_context("spawn")) as ex:
        futures = {}
        for t in tables:
            df_raw = t.get_df()
            # WICHTIG: nur einmal holen und sanitizen
            df = _sanitize_df_for_mp(df_raw)
            # nichts Pandas-/NumPy-spezifisches über die Prozess-Grenze schicken
            record_values = df.values.tolist()           # List[List[...]]
            record_columns = list(df.columns)            # List[str]
            fut = ex.submit(
                build_table_text_representation,
                t.table_name,
                columns[t.table_name],                  # Schema-Liste
                record_values,
                record_columns,
                foreign_keys,
                primary_keys
            )
            futures[fut] = t.table_name
        for fut in as_completed(futures):
            reps = fut.result()
            reps["sample_data"] = pd.DataFrame(reps["sample_data"])
            results[reps["table_name"]] = reps
            print(f"Processed {len(results)}/{total} tables", end="\r")
    return results

def _build_for_df(table_name: str, df: pd.DataFrame, k: int):
    print("Building representatives for", table_name)
    reps = get_representative_records_df(df, k)
    print(f"Finished building representatives for {table_name}")
    return table_name, reps


def build_table_text_representation(table_name, columns, record_values, record_columns, all_foreign_keys, primary_keys):
        data_samples = get_representative_records_df(record_values, record_columns, 5)
        # # data_samples = None
        if data_samples is None:
           print(f"No representative records for table {table_name}, using random sample instead.")
           data_samples = pd.DataFrame(record_values).sample(min(5, len(record_values)), random_state=1)
           #data_samples = table.get_df(5)
        # data_samples = table.get_df(5)
        
        #data_samples = table.get_df(5)#pd.DataFrame({col["name"]: table._get_data(col["name"], 5) for col in table.columns})
        fk_description = ""
        for fk in all_foreign_keys.get(table_name, []):
            fk_description += f" Foreign key '{' '.join(fk['constrained_columns'])}' references '{fk['referred_table']}'."
        #get foreign keys pointing to that table
        #fk_description += " Foreign keys pointing to this table: "

        #tables = table.db.get_tables()
        print("iterating over tables")
        for t, fks in all_foreign_keys.items():
            for fk in fks:
                if fk["referred_table"] == table_name:
                    fk_description += f" Table '{t}' has a foreign key '{' '.join(fk['constrained_columns'])}' pointing to this table."
        # for t in tables:
        #     for fk in t.get_foreign_keys():
        #         if fk["referred_table"] == table_name:
        #             fk_description += f" Table '{t.table_name}' has a foreign key '{' '.join(fk['constrained_columns'])}' pointing to this table."
        
        # database_context = "This table is part of the following database schema: "
        # for t in tables:
        #     if t.table_name == table_name:
        #         continue
        #     database_context += f"Table '{t.table_name}' "
        
        #primary_key = #", ".join(table.get_primary_key())
        primary_key = ", ".join(primary_keys.get(table_name, []))
        pd.set_option("display.max_columns", None)
        print(f"Done for table {table_name}")
        return {
            "table_name": table_name,
            "schema": columns,
            "sample_data": data_samples.to_dict(orient="records"),
            "fk_description": fk_description,
            #"database_context": database_context,
            "pk": primary_key
        }