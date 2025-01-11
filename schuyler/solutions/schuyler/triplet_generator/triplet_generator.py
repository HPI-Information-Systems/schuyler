from abc import ABC, abstractmethod
import wandb

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
