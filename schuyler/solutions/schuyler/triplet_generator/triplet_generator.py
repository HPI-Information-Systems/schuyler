from abc import ABC, abstractmethod
import wandb
from datasets import Dataset, DatasetDict
class BaseTripletGenerator(ABC):
    def __init__(self, database, groundtruth):
        self.database = database
        self.groundtruth = groundtruth

    @abstractmethod
    def generate_triplets(self, data_description):
        pass

    def analyze_triplet_selection(self, triplets):
        positive_in_same_cluster = 0
        negative_in_different_cluster = 0
        for el in triplets:
            in_same_cluster = False
            for cluster in self.groundtruth.clusters:
                if str(el[0]) in cluster and str(el[1]) in cluster:
                    positive_in_same_cluster += 1
                    in_same_cluster = True
                if str(el[0]) in cluster and str(el[2]) not in cluster:
                    negative_in_different_cluster += 1
                if str(el[0]) in cluster and str(el[2]) in cluster:
                    print("Negative element in same cluster", el[0], el[1], el[2])
            if not in_same_cluster:
                print("Positive element not in same cluster", el[0], el[1], el[2])
        wandb.log({"no_of_triplets": len(triplets), "positive_in_same_cluster": positive_in_same_cluster / len(triplets), "negative_in_different_cluster": negative_in_different_cluster / len(triplets)})

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

    def enrich_pairs(self, triplets):
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
        return dataset.shuffle(seed=42)