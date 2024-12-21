from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
import pandas as pd
import numpy as np
import statistics
import copy
from schuyler.solutions.iDisc.base_clusterer.BaseClusterer import BaseClusterer
from schuyler.solutions.iDisc.preprocessor import VectorRepresentator, SimilarityBasedRepresentator
def square_to_condensed(square):
    n = square.shape[0]
    return [square[i, j] for i in range(n) for j in range(i + 1, n)]
class SimilarityBasedClusterer(BaseClusterer):
    def __init__(self, representator, linkage='single', metric='cosine'):
        super().__init__()
        self.linkage = linkage
        self.metric = metric
        self.representator = representator

    def cluster(self, tables, dist=None):
        #self.representator = "VectorRepresentator" if isinstance(data, VectorRepresentator) else "SimilarityBasedRepresentator"
        if isinstance(self.representator, VectorRepresentator) or isinstance(self.representator, SimilarityBasedRepresentator):
            dist = self.representator.get_dist_matrix()
        #elif isinstance(self.representator, np.ndarray) or isinstance(self.representator, pd.DataFrame):
            #only allowed when meta clusterer
            #dist = self.representator
        elif dist is not None:
            dist = square_to_condensed(dist)
            print("Input from meta clusterer")
        else:
            raise ValueError("Invalid data type")
        return self._perform_base_clustering(tables=tables, distance_matrix=dist, linkage_method=self.linkage, metric=self.metric)
    
    def _perform_base_clustering(self, tables, distance_matrix, linkage_method='single', metric='cosine'):
        best_quality = float('-inf')
        best_clusters = None
        clusters_dict = {i: [i] for i in range(len(tables))}
        linkage_matrix = linkage(distance_matrix, method=linkage_method, metric=metric)
        for i in range(0, len(linkage_matrix)):
            print("Iteration", i)
            cluster_id = i + len(tables)
            row = linkage_matrix[i]
            new_cluster = clusters_dict[int(row[0])] + clusters_dict[int(row[1])]
            clusters_dict[cluster_id] = new_cluster
            del clusters_dict[int(row[0])]
            del clusters_dict[int(row[1])]
            print("Clusters", clusters_dict)
            quality = self.cluster_quality(clusters_dict, tables, squareform(distance_matrix))
            if quality > best_quality:
                best_quality = quality
                #copy clusters_dict
                best_clusters = copy.deepcopy(clusters_dict)
                #best_clusters = clusters_dict
                print(f"New best cluster quality with {quality}. Clusters: {best_clusters}")
        #convert index to table names
        output = []
        for cluster in best_clusters.keys():
            output.append([tables[i].table_name for i in best_clusters[cluster]])
            #best_clusters[cluster] = [tables[i] for i in best_clusters[cluster
        print("best clusters", best_clusters)
        print("Best output", output)
        #best_clusters = {tables[i].table_name: best_clusters[i] for i in best_clusters.keys()}
        return output
    
    def cluster_quality(self, clusters, tables, distance_matrix):
        clusters = [clusters[key] for key in clusters.keys()]
        return statistics.fsum([(len(clusters) * (self.intra_dist(cluster, clusters, distance_matrix) - self.inter_dist(cluster, distance_matrix))) / len(tables) \
                 for cluster in clusters])

    def inter_dist(self, cluster, distance_matrix):
        if len(cluster) == 1:
            return 1
        return statistics.mean([distance_matrix[i][j] for i in cluster for j in cluster if i != j])

    def cluster_similarity(self, cluster1, cluster2, distance_matrix):
        return np.mean([distance_matrix[i][j] for i in cluster1 for j in cluster2])

    def intra_dist(self, cluster, clustering, distance_matrix):
        if len(clustering) == 1:
            return 1
        return np.min([self.cluster_similarity(cluster, _cls, distance_matrix) for _cls in clustering if _cls != cluster])
        