from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, rand_score, adjusted_rand_score
import numpy as np


def mutual_information(true_database_cluster, pred_database_cluster, metric="") -> float:
    labels = np.array(true_database_cluster).flatten()
    true_labels = convert_to_labels(true_database_cluster, labels)
    pred_labels = convert_to_labels(pred_database_cluster, labels)
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