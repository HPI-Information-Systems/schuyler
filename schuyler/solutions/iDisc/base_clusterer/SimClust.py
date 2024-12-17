import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

from schuyler.solutions.iDisc.base_clusterer.BaseClusterer import BaseClusterer
from schuyler.solutions.iDisc.preprocessor import VectorRepresentor, SimilarityRepresentator

class SimilarityBasedClusterer(BaseClusterer):
    def __init__(self, linkage_method='single', metric='cosine'):
        super().__init__()
        self.linkage_method = linkage_method
        self.metric = metric

    def cluster(self, data: VectorRepresentor):
        if isinstance(data, VectorRepresentor):
            sim = pdist(data, metric='cosine')
        elif isinstance(data, SimilarityRepresentator):
            sim = data.get_representation()
        else:
            raise ValueError("Invalid data type")
        return self._perform_base_clustering(sim)
    
    def _perform_base_clustering(self, similarity_matrix, linkage_method='single', metric='cosine'):
        best_quality = float('-inf')
        best_clusters = None
        clusters = [[i] for i in range(similarity_matrix.shape[0])]
        linkage_matrix = linkage(clusters, method=linkage_method, metric=metric)
        for i in range(1, len(linkage_matrix)):
            cluster_id = i + similarity_matrix.shape[0]
            row = linkage_matrix[i]
            clusters[row[0]] = cluster_id
            clusters[row[1]] = cluster_id
            quality = self.cluster_quality(self.convert_to_cluster_representation(clusters))
            if quality > best_quality:
                best_quality = quality
                best_clusters = clusters
        return best_clusters
    
    def cluster_quality(self, clusters):
        return np.sum([(len(clusters) * (self.intra_sim(cluster) - self.inter_sim(cluster))) / len(self.similarity_matrix.shape[0]) \
                 for cluster in clusters])

    def inter_sim(self, cluster):
        return np.max([self.similarity_matrix[i][j] for i in cluster for j in cluster if i != j])

    def cluster_similarity(self, cluster1, cluster2):
        return np.mean([self.similarity_matrix[i][j] for i in cluster1 for j in cluster2])

    def intra_sim(self, cluster, clustering):
        return np.mean([self.cluster_similarity(cluster, _cls) for _cls in clustering if _cls != cluster])

    def convert_to_cluster_representation(self, cluster_structure):
        clusters = [[] for i in list(set(cluster_structure))]
        for i in range(len(cluster_structure)):
            clusters[cluster_structure[i]].append(i)
        return clusters
        


    