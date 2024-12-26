from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, rand_score, adjusted_rand_score
import numpy as np

from schuyler.experimenter.result import Result


def cluster_metric(true_clusters: Result, pred_clusters: Result, metric="") -> float:
    labels = true_clusters.get_labels()
    true_labels = true_clusters.convert_to_labels(labels)
    pred_labels = pred_clusters.convert_to_labels(labels)
    match metric:
        case "mutual_info_score":
            metric = mutual_info_score
        case "adjusted_mutual_info_score":
            metric = adjusted_mutual_info_score
        case "rand_score":
            metric = rand_score
        case "adjusted_rand_score":
            metric = adjusted_rand_score
        case _: 
            raise ValueError("Invalid metric")
    return metric(true_labels, pred_labels)

def convert_to_labels(clusters, labels):
    temp_clusters = []
    true_labels = np.zeros(len(labels))
    for cluster in clusters:
        temp_clusters.append([np.where(labels == table)[0][0] for table in cluster])
    for i, cluster in enumerate(temp_clusters):
        for table in cluster:
            true_labels[table] = i
    return true_labels