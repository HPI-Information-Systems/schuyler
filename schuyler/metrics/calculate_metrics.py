from schuyler.metrics.MutualInformation import cluster_metric
from schuyler.experimenter.result import Result


def calculate_metrics(output: Result, groundtruth: Result):
    metrics = ["mutual_info_score", "adjusted_mutual_info_score", "rand_score", "adjusted_rand_score"]
    results = {}

    for metric in metrics:
        results[metric] = cluster_metric(groundtruth, output, metric)

    #calculate statistics about the clusters and compare it to groundtruth
    output_statistics = cluster_statistics(output.clusters)
    groundtruth_statistics = cluster_statistics(groundtruth.clusters)
    for metric in output_statistics:
        results[f"output_{metric}"] = output_statistics[metric]
    for metric in groundtruth_statistics:
        results[f"groundtruth_{metric}"] = groundtruth_statistics[metric]

    return results

def cluster_statistics(clusters):
    cluster_stats = {}
    cluster_stats["number_of_clusters"] = len(clusters)
    cluster_stats["average_cluster_size"] = sum([len(cluster) for cluster in clusters])/len(clusters)
    cluster_stats["max_cluster_size"] = max([len(cluster) for cluster in clusters])
    cluster_stats["min_cluster_size"] = min([len(cluster) for cluster in clusters])
    return cluster_stats


def hierarchical_calculate_metrics(output: Result, groundtruth: Result):
    
    groundtruth = groundtruth.load_groundtruth(groundtruth.path)


    pass