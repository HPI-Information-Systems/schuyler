from schuyler.metrics.MutualInformation import cluster_metric

def calculate_metrics(output, groundtruth):
    metrics = ["mutual_info_score", "adjusted_mutual_info_score", "rand_score", "adjusted_rand_score"]
    results = {}

    for metric in metrics:
        results[metric] = cluster_metric(groundtruth, output, metric)
    return results