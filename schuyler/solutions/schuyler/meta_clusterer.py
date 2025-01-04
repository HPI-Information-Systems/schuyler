import pandas as pd
import networkx as nx
from sklearn.cluster import AffinityPropagation


class MetaClusterer:
    def __init__(self, graph):
        self.graph = graph
    
    def cluster(self, clusterings, alpha=0.5):
        if len(clusterings) != 2:
            raise ValueError("This method expects exactly two clusterings as input.")
            
        nodes = self.graph.nodes
        tables = [node.table.table_name for node in nodes]
        df = pd.DataFrame(columns=tables, index=tables)
        
        for node1 in nodes:
            for node2 in nodes:
                in_same_cluster_1 = self.in_same_cluster(node1.table.table_name, node2.table.table_name, clusterings[0])
                in_same_cluster_2 = self.in_same_cluster(node1.table.table_name, node2.table.table_name, clusterings[1])
                weighted_score = alpha * in_same_cluster_1 + (1 - alpha) * in_same_cluster_2
                
                df.loc[node1.table.table_name, node2.table.table_name] = weighted_score
                df.loc[node2.table.table_name, node1.table.table_name] = weighted_score
        
        print(df)
        df.to_csv("/data/adj.csv")
        G = nx.from_pandas_adjacency(df.astype(float))
        
        similarity_matrix = nx.to_numpy_array(G)
        
        affinity_propagation = AffinityPropagation(affinity='precomputed', random_state=0)
        affinity_propagation.fit(similarity_matrix)
        
        cluster_labels = affinity_propagation.labels_
        
        clusters = {}
        for node, label in zip(nodes, cluster_labels):
            table_name = node.table.table_name
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(table_name)
        
        return list(clusters.values())

    
    def in_same_cluster(self, table1, table2, clustering): return any(table1 in cluster and table2 in cluster for cluster in clustering)