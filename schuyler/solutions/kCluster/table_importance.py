from schuyler.database.database import Database
import numpy as np
from scipy.stats import entropy
import networkx as nx
from scipy.sparse.linalg import eigs
import pandas as pd

class TableImportance:
    def __init__(self, database: Database):
        self.database = database
        entropy_dict = self.calculate_entropy() #verified
        #self.graph, self.transfer_matrix = self.construct_entropy_transfer_matrix()
        self.prob_matrix = self.construct_probability_matrix(entropy_dict)
        
        database_name = self.database.database.split('__')[1]
        # self.prob_matrix = pd.read_csv(f"/data/{database_name}/prob_matrix.csv", index_col=0)
        self.stationary_dist = self.eigen(self.prob_matrix)
        # self.stationary_dist = self.compute_stationary_distribution(self.prob_matrix)
        
    def calculate_entropy(self):
        tables = self.database.get_tables()
        entropy_dict = {}
        for table in tables:
            df = table.get_df()
            print(df.info())
            attr_entropy_dict = {}
            for col in df.columns:
                attr_entropy_dict[col] = entropy(df[col].value_counts(), base=2)
            entropy_dict[table.table_name] = attr_entropy_dict
            entropy_dict[table.table_name]["pk"] = np.log2(len(df))
        print("Entropy dict:")
        print(entropy_dict)
        return entropy_dict
            #table_entropy /= len(df.columns)     

    def compute_stationary_distribution(self, df_prob_matrix, tol=1e-06, max_iter=1000):
        P = df_prob_matrix.values
        n = P.shape[0]
        pi = np.ones(n) / n
        for iteration in range(max_iter):
            next_pi = np.dot(pi, P)
            #next_pi /= np.sum(next_pi)
            if np.max(np.abs(next_pi - pi)) <= tol:
                print(f"Converged in {iteration + 1} iterations.")
                df = pd.DataFrame(next_pi, index = df_prob_matrix.index)
                df.sort_values(by=0, ascending=False, inplace=True)
                df.to_csv("/data/stationary_dist_iterative_2.csv")
                print(df)
                return next_pi
            pi = next_pi
        print("Stationary distribution:", pi)
        print(iteration)
        return pi
    

    def eigen(self, prob_matrix_df):
        print(prob_matrix_df.index)
        print(prob_matrix_df.columns)
        prob_matrix = prob_matrix_df.values
        for method in ["LM"]:#, "SM", "LR", "SR", "LI", "SI"]:
            eigenvalues, eigenvectors = eigs(prob_matrix.T, k=1, which=method)
            stationary_dist = eigenvectors[:, 0].real
            stationary_dist /= stationary_dist.sum()  # Normalize to sum to 1
            print("Stationary distribution:", stationary_dist)
        #dump to csv with table names as index
            columns = [table.table_name for table in self.database.get_tables()]
            df = pd.DataFrame(stationary_dist, index = columns)
            df.sort_values(by=0, ascending=False, inplace=True)
            print(method)
            print(df)
        print(df)
        df.to_csv("/data/stationary_dist.csv")
        return stationary_dist
    
    def pr(self, attribute_a, table_a, attribute_b, table_b, entropy_dict):
        print(f"Calculating pr for {attribute_a} in {table_a.table_name} and {attribute_b} in {table_b}")
        entropy_a = entropy_dict[table_a.table_name][attribute_a]
        table_entropy = np.sum(list(entropy_dict[table_a.table_name].values()))
        print(f"Table entropy of {table_a.table_name}: {table_entropy}")
        amount_of_fks_where_attribute_a_is_included = 0
        fks = self.database.get_foreign_keys()
        amount_of_fks_where_attribute_a_is_included = len([fk for fk in fks if fk["constrained_table"] == table_a.table_name and attribute_a in fk["constrained_columns"]]) #or attribute_a in fk["referred_columns"] and fk["referred_table"] == table_a])
        print(f"I Amount of fks where {attribute_a} is included: {amount_of_fks_where_attribute_a_is_included}")
        # for table in self.database.get_tables():
        #     for fk in table.get_foreign_keys():
        #         if fk["referred_table"] == table_a.table_name:
        #             print("FK")
        #             print(table_a.table_name)
        #             print(attribute_a)
        #             print(fk)
                
        #         if fk["referred_table"] == table_a.table_name and attribute_a in fk["constrained_columns"] or attribute_a in fk["referred_columns"]:
        #             amount_of_fks_where_attribute_a_is_included += 1
        print(f"Amount of fks where {attribute_a} is included: {amount_of_fks_where_attribute_a_is_included}")
        sum = 0
        cols = list(table_a.get_df().columns)
        for attr in cols: #todo most likely muss hier das - attribute_a weg
            sum += (amount_of_fks_where_attribute_a_is_included - 1) * entropy_dict[table_a.table_name][attr]
        print(f"Entropy of {attribute_a} in {table_a.table_name}: {entropy_a}")
        return entropy_a / (table_entropy + sum)

    def construct_probability_matrix(self, entropy_dict):
        tables = [table.table_name for table in self.database.get_tables()]
        df = pd.DataFrame(columns = tables, index = tables)
        df = df.applymap(lambda x: 0)
        ic_tables = []
        table_objects = self.database.get_tables()
        for table in table_objects:
            ic = np.sum(list(entropy_dict[table.table_name].values()))
            ic_tables.append(ic)
            fks = table.get_foreign_keys()
            for fk in fks:
                table_a = table
                table_b = fk["referred_table"]
                attribute_a = fk["constrained_columns"][0]
                attribute_b = fk["referred_columns"][0]
                #assert len(attribute_a) == 1 and len(attribute_b) == 1 #todo paper seems to only support single node fks
                if table_b == table.table_name:
                    print(f"Table {table_a} has a foreign key to itself. Will be handled later.")
                    continue
                df[table_b][table.table_name] += self.pr(attribute_a, table_a, attribute_b, table_b, entropy_dict)
                #df[table_b][table.table_name] += self.pr(attribute_b, table_b, attribute_a, table_a, entropy_dict)
            # for fk in fks:
            #     if table_b == table.table_name:
            #         print(f"Table {table_a} has a foreign key to itself. Will be handled now.")
            #         df[table.table_name][table_b] = 1 - df[table.table_name].sum()
            # if table.table_name == "account_permission":
            #     raise ValueError
        for table in self.database.get_tables():
            df[table.table_name][table.table_name] = 1 - df.loc[table.table_name].sum()
        print("Probability matrix:")
        print(df.sum(axis=1))
        database_name = self.database.database.split('__')[1]
        ic_tables = pd.DataFrame(ic_tables, index = [table_object.table_name for table_object in table_objects])
        ic_tables.to_csv(f"/data/{database_name}/ic_tables.csv")
        df.to_csv(f"/data/{database_name}/prob_matrix.csv")
        return df

    def construct_entropy_transfer_matrix(self):
        nodes = self.database.get_tables()
        edges = []
        for node in nodes:
            for fk in node.get_foreign_keys():
                edges.append((node, fk["referred_table"]))
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        transfer_matrix = nx.to_numpy_array(graph)
        return graph, transfer_matrix
        # transfer_matrix = np.zeros((len(nodes), len(nodes)))
        # for edge in edges:
        #     transfer_matrix[nodes.index(edge[0]), nodes.index(edge[1])] = 1
        # return transfer_matrix