from schuyler.database.database import Database
from schuyler.experimenter.result import Result

def foreign_key_cluster_belongness(database: Database, groundtruth: Result):
    fks = database.get_foreign_keys()
    clusters = groundtruth.clusters
    #get foreign keys that refer to tables being in the same cluster
    same_cluster_fks = 0
    for table in database.get_tables():
        for fk in table.get_foreign_keys():
            for cluster in clusters:
                if table.table_name in cluster and fk["referred_table"] in cluster:
                    same_cluster_fks += 1
                    break
    return same_cluster_fks / len(fks)

