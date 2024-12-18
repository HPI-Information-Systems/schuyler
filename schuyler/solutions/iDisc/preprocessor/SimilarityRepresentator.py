import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from schuyler.solutions.iDisc.preprocessor.BaseRepresentator import BaseRepresentator
from schuyler.database import Database, Table

class SimilarityBasedRepresentator(BaseRepresentator):
    def __init__(self, database: Database):
        super().__init__(database)

    def get_representation(self):
        tables = self.database.get_tables()
        tab_sim = pd.DataFrame(index=[t.table_name for t in tables], columns=[t.table_name for t in tables])
        np.fill_diagonal(tab_sim.values, 1)
        for table1 in tables:
            for table2 in tables:
                if table1 == table2:
                    continue
                sim = self.greedy_attribute_matching(self.table_similarity(table1, table2))
                tab_sim.loc[table1.table_name, table2.table_name] = sim
                tab_sim.loc[table2.table_name, table1.table_name] = sim
        return tab_sim
    
    def greedy_attribute_matching(self, similarity_matrix):
        """
            Greedy attribute matching based on similarity matrix
        """
        similarity_matrix = similarity_matrix.copy()
        result = []
        while not similarity_matrix.empty:
            max_pair = similarity_matrix.stack().idxmax()
            sim = similarity_matrix.loc[max_pair[0], max_pair[1]]
            result.append(sim)
            similarity_matrix = similarity_matrix.drop(index=max_pair[0],columns=max_pair[1])
        return np.mean(result)


        

    def table_similarity(self, table1: Table, table2: Table):
        """
            Table similarity by jaccard distance per attribute pair (cross product)
        """
        data_table1 = {c['name']: set(table1._get_data(c['name'], 200)) for c in table1._get_columns()} #todo make unqiue values
        data_table2 = {c['name']: set(table2._get_data(c['name'], 200)) for c in table2._get_columns()}
        unique_values = set().union(*data_table1.values(), *data_table2.values())
        def encode_binary(column_set, unique_values):
            return [1 if value in column_set else 0 for value in unique_values]
        binary_matrix_a = [encode_binary(set_a, unique_values) for set_a in data_table1.values()]
        binary_matrix_b = [encode_binary(set_b, unique_values) for set_b in data_table2.values()]
        distances = cdist(binary_matrix_a, binary_matrix_b, metric='jaccard')
        similarities = 1 - distances
        similarity_matrix = pd.DataFrame(
            similarities, index=data_table1.keys(), columns=data_table2.keys())
        
        return similarity_matrix