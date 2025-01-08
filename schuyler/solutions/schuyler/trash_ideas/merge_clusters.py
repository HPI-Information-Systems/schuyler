cluster_sizes = [len(cluster) for cluster in clustering]
print("Cluster sizes", cluster_sizes)
#average cluster size of lower quartile
percentile = np.percentile(cluster_sizes, 20)

print("Percentile", percentile)
clusters_to_merge = [cluster for cluster in clustering if len(cluster) <= percentile]
print("Clusters to merge", clusters_to_merge)
cluster_embeddings = []
print("Calculating cluster embeddings")
for cluster in clustering:
    print(cluster)
    cluster_embeddings.append(calculate_cluster_embedding(cluster, G))
for i, cluster in enumerate(clustering):
    if len(cluster) <= percentile:
        print("Merging cluster", cluster)
        cluster_embedding = cluster_embeddings[i]
        most_similar_cluster = clustering[select_most_similar_cluster(cluster_embedding=cluster_embedding, reference_cluster_embeddings=cluster_embeddings)]
        # add to cluster directly in clustering
        most_similar_cluster.extend(cluster)
        clustering.remove(cluster)
        print("Merged cluster", cluster, "into cluster", most_similar_cluster)
print("OUTPUT", clustering)

print("upper", np.percentile(cluster_sizes, 90))