import numpy as np
import pandas as pd

from schuyler.solutions.iDisc.base_clusterer import BaseClusterer, SimilarityBasedClusterer

class MetaClusterer:
    def __init__(self, database):
        self.database = database
        self.representator = "meta"

    def cluster(self, clusterings):
        print("CLUSTERING", clusterings)
        tables =self.database.get_tables()
        table_names = list(map(lambda t: t.table_name, tables))
        #tables = list(map(lambda t: t.table_name, ))
        table_sim = pd.DataFrame(index=table_names, columns=table_names)
        table_sim.fillna(0, inplace=True)
        table_sim.values[[range(len(tables))]*2] = 1
        for table_1 in table_names:
            # print("TABLE 1", table_1)
            for table_2 in table_names:
                # print("TABLE 2", table_2)
                if table_1 != table_2:
                    sim = np.mean([self.in_same_cluster(table_1, table_2, clustering) for clustering in clusterings])
                    table_sim.loc[table_1, table_2] = sim
                    table_sim.loc[table_2, table_1] = sim
        print("SIM MATRIX", 1-table_sim.values)
        return SimilarityBasedClusterer(None, linkage="single").cluster(dist=1-table_sim.values, tables=tables)
        
    def in_same_cluster(self, table_1, table_2, clustering):
        # print("IN SAME CLUSTER", table_1, table_2, clustering)
        return any(map(lambda cluster: table_1 in cluster and table_2 in cluster, clustering))       
        
        