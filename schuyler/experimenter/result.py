import yaml
import numpy as np

class Result:
    def __init__(self, data=None, path=None):
        if path:
            self.load_groundtruth(path)  
        else:
            self.clusters = data

    def load_groundtruth(self, path):
        with open(path, 'r') as file:
            self.clusters = list(yaml.load(file, Loader=yaml.FullLoader)["clusters"].values())

    def convert_to_labels(self, labels):
        temp_clusters = []
        true_labels = np.zeros(len(labels))
        for cluster in self.clusters: #todo does it work
            print("hddhdhdhdhdhdh", cluster)
            print("dhbab", labels)
            temp_clusters.append([np.where(np.array(labels) == table)[0][0] for table in cluster])
        for i, cluster in enumerate(temp_clusters):
            for table in cluster:
                true_labels[table] = i
        return true_labels

    def get_labels(self) -> list[str]:
        print("clusters", self.clusters)
        return np.array(np.hstack(self.clusters))