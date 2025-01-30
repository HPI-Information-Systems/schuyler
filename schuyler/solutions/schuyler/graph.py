from schuyler.database.database import Database
from networkx import Graph, pagerank, betweenness_centrality
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.feature_vector.llm import LLM, SentenceTransformerModel
from schuyler.solutions.iDisc.preprocessor.SimilarityRepresentator import SimilarityBasedRepresentator
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import wandb



import os
import pandas as pd
from tqdm import tqdm
import sys
import pickle
from schuyler.solutions.iDisc.preprocessor.VectorRepresentator import VectorRepresentator
from schuyler.solutions.iDisc.preprocessor.document_builder.attribute_values import AttributeValuesDocumentBuilder
from schuyler.solutions.iDisc.preprocessor.document_builder.table_name_and_cols import TableNameAndColsDocumentBuilder
class DatabaseGraph:
    def __init__(self, database: Database, model=SentenceTransformerModel):
        self.graph = Graph()
        self.database = database
        self.llm = LLM()
        self.model = model()

    def construct(self, similar_table_connection_threshold=0.0, groundtruth=None):
        print("Constructing database graph...")
        print("Adding nodes...")
        self.nodes = [Node(table, llm=self.llm, model=self.model, groundtruth_label=groundtruth.get_label_for_table(table.table_name)) for table in self.database.get_tables()]
        # sim = VectorRepresentator(self.database, AttributeValuesDocumentBuilder).get_dist_matrix()
        self.graph.add_nodes_from(self.nodes)
        print("Adding edges...")
        # tfidf_sim = self.get_tfidf_similarity()
        for node1 in self.nodes:
            table = node1.table
            print(table.table_name)
            print(table.get_foreign_keys())
            for fk in table.get_foreign_keys():
                edge = Edge(node1, self.get_node(fk["referred_table"]), self.model)
                if edge.table_sim < 0.5:
                    print("Table similarity too low", edge, edge.table_sim)
                    continue
                self.graph.add_edge(edge.node1, edge.node2)
                self.graph[edge.node1][edge.node2]["edge"] = edge
                # if use_tfidf:
                #     self.graph[edge.node1][edge.node2]["weight"] = tfidf_sim.loc[table.table_name, fk["referred_table"]]
                # else:
                self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
                #self.graph[edge.node1][edge.node2]["weight"] = tfidf_sim.loc[table.table_name, fk["referred_table"]]
            # for node2 in self.nodes:
            #     if node1 == node2:
            #         continue
            #     edge = Edge(node1, node2, self.sentencetransformer)
            #     self.graph.add_edge(edge.node1, edge.node2)
            #     self.graph[edge.node1][edge.node2]["edge"] = edge
            #     self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
        print("Calculating pagerank")
        pr = pagerank(self.graph, alpha=0.85)
        print("Calculating betweenness centrality")
        b = betweenness_centrality(self.graph, normalized=True, endpoints=False, seed=42)
        print("Calculating features")
        for node in tqdm(self.nodes, file=sys.stdout):
            print(node.table.table_name)
            node_features_file = f"/data/{node.table.db.database.split('__')[1]}/results/nodes/{node.table.table_name}.pkl"
            os.makedirs(f"/data/{node.table.db.database.split('__')[1]}/results/nodes", exist_ok=True)
            if os.path.exists(node_features_file):
                print("Graph file exists")
                with open(node_features_file, "rb") as f:
                    node_features = pickle.load(f)
                node.page_rank = node_features["page_rank"]
                node.degree = node_features["degree"]
                node.betweenness_centrality = node_features["betweenness_centrality"]
                node.embeddings = node_features["embeddings"]
                node.features = node_features["features"]
                node.average_semantic_similarity = node_features["features"][3]
                node.amount_of_fks = node_features["features"][4]
                node.amount_of_columns = node_features["features"][5]
                node.row_count = node_features["features"][6]
            else:
                node.page_rank = pr[node]
                node.degree = self.graph.degree(node)
                node.betweenness_centrality = b[node]
                v = node.calculate_feature_vector(self.graph)
                node.embeddings = v["embeddings"].tolist()
                node.average_semantic_similarity = v["average_semantic_similarity"]
                node.amount_of_fks = v["amount_of_fks"]
                node.amount_of_columns = v["amount_of_columns"]
                node.row_count = v["row_count"]
                node.features = [node.page_rank, node.degree, node.betweenness_centrality, v["average_semantic_similarity"], v["amount_of_fks"], v["amount_of_columns"], v["row_count"]]
                with open(node_features_file, "wb") as f:
                    pickle.dump({"page_rank": node.page_rank, "degree": node.degree, "betweenness_centrality": node.betweenness_centrality, "embeddings": node.embeddings, "features": node.features}, f)
            print(f"Node {node.table.table_name} has a page rank of {node.page_rank}, a degree of {node.degree}, a betweenness centrality of {node.betweenness_centrality}")
        similar_tables = self.get_similar_embedding_tables(similar_table_connection_threshold)
        if similar_table_connection_threshold > 0.0:
            print("Adding similar table connections")
            # similar_tables_1 = self.get_similar_value_tables(similar_table_connection_threshold)
            similar_tables = self.get_similar_tfidf_tables(similar_table_connection_threshold)
            # similar_tables = list(set(similar_tables_1).union(set(similar_tables_2)))

            for table1, table2, sim in similar_tables:
                node1 = self.get_node(table1)
                node2 = self.get_node(table2)
                edge = Edge(node1, node2, self.model, sim)
                self.graph.add_edge(edge.node1, edge.node2)
                self.graph[edge.node1][edge.node2]["edge"] = edge

                self.graph[edge.node1][edge.node2]["weight"] = sim
                #self.graph[edge.node1][edge.node2]["weight"] = tfidf_sim.loc[table1, table2]

        print(self.graph.edges)
        print("Database graph constructed.")
        return self.graph

    def get_similar_value_tables(self, threshold):
        sim_matrix: pd.DataFrame = SimilarityBasedRepresentator(self.database).get_representation()
        #get all tables with similarity above threshold
        similar_tables = []
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix.iloc[i, j] > threshold:
                    similar_tables.append((sim_matrix.index[i], sim_matrix.columns[j]))
                    print("Somilar table connection", sim_matrix.index[i], sim_matrix.columns[j], sim_matrix.iloc[i, j])
        return similar_tables
    
    def visualize_embeddings(self, name):
        embeddings = []
        labels = []
        tabels = []

        for node, data in self.graph.nodes(data=True):
            embeddings.append(node.encoding)
            labels.append(node.groundtruth_label)
            tabels.append(node.table.table_name)
        #print(embeddings)
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        df = pd.DataFrame(embeddings)
        df['label'] = labels
        df['table'] = tabels
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_umap = umap_reducer.fit_transform(embeddings)
        print(embeddings)
        print(labels)
        sil_score = silhouette_score(embeddings, labels)
        db_index = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        wandb.log({f"silhouette_score_{name}": sil_score})
        wandb.log({f"davies_bouldin_score_{name}": db_index})
        wandb.log({f"calinski_harabasz_score_{name}": ch_score})

        plot_df = pd.DataFrame({
            'Dim1': embeddings_umap[:, 0],
            'Dim2': embeddings_umap[:, 1],
            'Label': labels,
            'Table': tabels
        })

        fig = px.scatter(
            plot_df,
            x='Dim1',
            y='Dim2',
            color='Label',
            title='Node Embeddings Visualization',
            labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'},
            hover_data=['Label', 'Table']
        )

        wandb.log({f"embedding_plot_{name}": fig})

    def get_similar_tfidf_tables(self, threshold):
        print("Calculating tfidf similarity")
        df = self.get_tfidf_similarity()
        similar_tables = []
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                if df.iloc[i, j] > threshold:
                    similar_tables.append((df.index[i], df.columns[j]))
                    # print("Similar table connection", df.index[i], df.columns[j], df.iloc[i, j])
        return similar_tables

    def update_encodings(self):
        for node in self.nodes:
            node.update_encoding(self.model)
        for edge in self.graph.edges:
            edge = self.graph[self.get_node(str(edge[0]))][self.get_node(str(edge[1]))]["edge"] 
            self.graph[edge.node1][edge.node2]["weight"] = edge.get_table_similarity()
            self.graph[edge.node1][edge.node2]["edge"].sim = edge.get_table_similarity()
            

    def get_tfidf(self):
        tables = self.database.get_tables()
        #df = pd.DataFrame(columns=[t.table_name for t in tables])

        rep = VectorRepresentator(self.database, AttributeValuesDocumentBuilder).get_representation()
        tfidf = {
        }
        for i, table in enumerate(tables):
            print(rep[i].toarray())
            tfidf[table.table_name] = rep[i].toarray()[0]
        #df = pd.DataFrame(rep.toarray(), columns=rep.get_feature_names_out(), index=[t.table_name for t in tables])
        # for i, table in enumerate(tables):
        #     df.loc[table.table_name] = rep[i].toarray()[0]
        return tfidf

    def get_tfidf_similarity(self):
        sim = 1 - VectorRepresentator(self.database, TableNameAndColsDocumentBuilder).get_dist_matrix()
        sim = squareform(sim)
        #print(np.shape(sim))
        tables = self.database.get_tables()
        df = pd.DataFrame(sim, index=[t.table_name for t in tables], columns=[t.table_name for t in tables])
        return df

    def get_similar_embedding_tables(self, threshold):
        print("Calculating embedding similarity")
        tables = self.database.get_tables()
        sim_matrix = pd.DataFrame(index=[t.table_name for t in tables], columns=[t.table_name for t in tables])
        similar_tables = []
        if not os.path.exists(f"/data/{self.database.database.split('__')[1]}/sim_matrix.csv"):
            for table1 in tqdm(tables):
                table1 = table1.table_name
                for table2 in tables:
                    table2 = table2.table_name
                    if table1 == table2:
                        continue
                    edge = Edge(self.get_node(table1), self.get_node(table2), self.model)
                    sim = edge.table_sim
                    sim_matrix.loc[table1, table2] = sim
                    if sim > threshold:
                        # print("Similar table connection", table1, table2, sim)
                        similar_tables.append((table1, table2))
        else:
            sim_matrix = pd.read_csv(f"/data/{self.database.database.split('__')[1]}/sim_matrix.csv", index_col=0)
            for table1 in tables:
                table1 = table1.table_name
                for table2 in tables:
                    table2 = table2.table_name
                    if table1 == table2:
                        continue
                    sim = sim_matrix.loc[table1, table2]
                    if sim > threshold:
                        # print("Similar table connection", table1, table2, sim)
                        similar_tables.append((table1, table2, sim))
        database_name = self.database.database.split("__")[1]
        sim_matrix.to_csv(f"/data/{database_name}/sim_matrix.csv")
        return similar_tables

    
    def get_node(self, table_name):
        if type(table_name) == Node:
            return table_name
        for n in self.graph.nodes:
            if n.table.table_name == table_name:
                return n
        return None
    


    