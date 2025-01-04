import yaml
import numpy as np

class Result:
    def __init__(self, hierarchy_level=0, data=None, path=None):
        self.hierarchy_level = hierarchy_level
        self.data = data
        self.path = path
        if path:
            self.load_groundtruth(path, hierarchy_level)  
        else:
            self.clusters = data

    def get_max_depth(self, d, current_depth=0):
        if isinstance(d, dict):
            if not d:
                return current_depth
            return max(self.get_max_depth(v, current_depth + 1) for v in d.values())
        else:
            return current_depth

    def load_groundtruth(self, path, level):
        with open(path, 'r') as file:
            yaml_dict = yaml.load(file, Loader=yaml.FullLoader)["clusters"]
        max_depth = self.get_max_depth(yaml_dict)
        if level > max_depth:
            print(f"Specified hierarchy level {level} exceeds maximum depth {max_depth}.")
            level = max_depth
        if level == 0:
            def merge_lists(d):
                merged = {}
                for key, value in d.items():
                    if isinstance(value, dict):
                        merged[key] = merge_lists(value)
                    elif isinstance(value, list):
                        merged.setdefault(key, []).extend(value)
                return merged
            def flatten_dict_to_list(d):
                result = []
                def recursive_flatten(sub_dict):
                    for value in sub_dict.values():
                        if isinstance(value, dict):
                            recursive_flatten(value)
                        else:
                            result.append(value)
                recursive_flatten(d)
                return result
            def flatten_top_level(d):
                flattened = {}
                for key, value in d.items():
                    if isinstance(value, dict):
                        flattened[key] = flatten_dict_to_list(value)
                    elif isinstance(value, list):
                        flattened[key] = value
                return flattened

            modified_dict = flatten_top_level(yaml_dict)
            # flatten lists of lists of each key to flat lists
            for key, value in modified_dict.items():
                if isinstance(value[0], list):
                    modified_dict[key] = [item for sublist in value for item in sublist]
        else:
            modified_dict = self.flatten_and_prefix(yaml_dict, target_level=level)
        self.clusters = list(modified_dict.values())
        

    def convert_to_labels(self, labels):
        temp_clusters = []
        true_labels = np.zeros(len(labels))
        for cluster in self.clusters: #todo does it work
            temp_clusters.append([np.where(np.array(labels) == table)[0][0] for table in cluster])
        for i, cluster in enumerate(temp_clusters):
            for table in cluster:
                true_labels[table] = i
        return true_labels

    def get_labels(self):
        return np.array(np.hstack(self.clusters))
    
    def flatten_and_prefix(self, d, prefix='', target_level=1, current_level=0):
        flat_dict = {}
        if isinstance(d, dict):
            for key, value in d.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                if current_level < target_level:
                    if isinstance(value, dict):
                        deeper = self.flatten_and_prefix(value, new_prefix, target_level, current_level + 1)
                        flat_dict.update(deeper)
                    else:
                        flat_dict[new_prefix] = value
                elif current_level == target_level:
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            combined_key = sub_key#f"{new_prefix}_{sub_key}"
                            if isinstance(sub_value, dict):
                                deeper = self.flatten_and_prefix(sub_value, combined_key, target_level, current_level + 1)
                                flat_dict.update(deeper)
                            else:
                                flat_dict[combined_key] = sub_value
                    else:
                        flat_dict[new_prefix] = value
        else:
            flat_dict[prefix] = d
        return flat_dict