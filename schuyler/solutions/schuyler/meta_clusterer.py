import pandas as pd
import networkx as nx

class MetaClusterer:
    def __init__(self, graph):
        self.graph = graph
    
    def cluster(self, clusterings):
        nodes = self.graph.nodes
        tables = [node.table.table_name for node in nodes]
        df = pd.DataFrame(columns=tables, index=tables)
        for node1 in nodes:
            for node2 in nodes:
                # count occurences of table1 and table2 in same cluster
                count = 0
                for clustering in clusterings:
                    if self.in_same_cluster(node1.table.table_name, node2.table.table_name, clustering):
                        count += 1
                df.loc[node1.table.table_name, node2.table.table_name] = count / len(clusterings)
                df.loc[node2.table.table_name, node1.table.table_name] = count / len(clusterings)
        #convert adjecency matrix to graph
        print(df)
        df.to_csv("/data/adj.csv")
        G = nx.from_pandas_adjacency(df.astype(float))
        clusters = nx.community.louvain_communities(G)
        return clusters
    
    def in_same_cluster(self, table1, table2, clustering): return any(table1 in cluster and table2 in cluster for cluster in clustering)