from schuyler.database import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
#import kmeans


def get_representative_records(table, amount_of_records=5):
    records = table.get_df()
    print(records)
    if records.empty or len(records) < amount_of_records:
        return None
    print(records)
    record_values = records.values
    documents = [" ".join(map(str, record)) for record in record_values]

    print("Calculating representative records")
    #print(documents)
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(documents)
    except ValueError:
        return None
    print("Fitting KMeans")
    kmeans = KMeans(n_clusters=amount_of_records, random_state=0)
    kmeans.fit(X)
    print("KMeans finished")

    representatives = []
    for cluster_idx in range(amount_of_records):
        print(kmeans.labels_)
        cluster_indices = np.where(kmeans.labels_ == cluster_idx)[0]
        if len(cluster_indices) == 0:
            continue
        print("X", X)
        print("ci", cluster_indices)
        cluster_vectors = X[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_idx]
        print("cc", centroid)
        print("cv", cluster_vectors)
        similarities = cosine_similarity(cluster_vectors, centroid.reshape(1, -1))
        closest_idx = cluster_indices[np.argmax(similarities)]
        representatives.append(record_values[closest_idx])

    representative_df = pd.DataFrame(representatives, columns=records.columns)
    return representative_df
