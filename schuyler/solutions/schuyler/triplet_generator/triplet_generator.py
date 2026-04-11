from abc import ABC, abstractmethod
import wandb
from datasets import Dataset
from collections import defaultdict
from itertools import combinations
class BaseTripletGenerator(ABC):
    def __init__(self, database):
        self.database = database

    @abstractmethod
    def generate_triplets(self, data_description):
        pass

    def find_overlapping_elements(self,clusters):
        """
        Returns a dict mapping each element to the list of cluster indices it appears in,
        for elements that appear in more than one cluster.
        """
        element_to_clusters = defaultdict(list)
        for idx, cluster in enumerate(clusters):
            for elm in cluster:
                element_to_clusters[str(elm)].append(idx)
        overlaps = {elm: idxs for elm, idxs in element_to_clusters.items() if len(idxs) > 1}
        return overlaps

    def find_overlapping_cluster_pairs(self,clusters):
        """
        Returns a list of tuples (i, j, intersection_set) for each pair of clusters
        that have a non-empty intersection.
        """
        overlaps = []
        for i, j in combinations(range(len(clusters)), 2):
            inter = set(map(str, clusters[i])) & set(map(str, clusters[j]))
            if inter:
                overlaps.append((i, j, inter))
        return overlaps


    def enrich_triplets(self, triplets):
        anchors = [anchor.llm_description for anchor, _, _ in triplets]
        positives = [positive.llm_description for _, positive, _ in triplets]
        negatives = [negative.llm_description for _, _, negative in triplets]
        data = {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        }
        return Dataset.from_dict(data)
    
    def convert_to_pairs(self, triplets):
        pairs = []
        for anchor, positive, negative in triplets:
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
        return pairs

    def enrich_pairs(self, triplets, seed=42):
        anchors = [anchor.llm_description for anchor, _, _ in triplets]
        positives = [positive.llm_description for _, positive, _ in triplets]
        negatives = [negative.llm_description for _, _, negative in triplets]
        labels = [1] * len(anchors) + [0] * len(anchors)
        data = {
            "sentence1": anchors + anchors,
            "sentence2": positives + negatives,
            "label": labels
        }
        dataset = Dataset.from_dict(data)
        return dataset.shuffle(seed=seed)