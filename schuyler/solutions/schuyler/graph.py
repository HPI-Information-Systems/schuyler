from schuyler.database.database import Database
from networkx import Graph, pagerank, betweenness_centrality
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.feature_vector.llm import LLM, SentenceTransformerModel
import os
from tqdm import tqdm
import sys
import pickle
from schuyler.solutions.iDisc.preprocessor.VectorRepresentator import VectorRepresentator
from schuyler.solutions.iDisc.preprocessor.document_builder.attribute_values import AttributeValuesDocumentBuilder
class DatabaseGraph:
    def __init__(self, database: Database):
        self.graph = Graph()
        self.database = database
        self.llm = LLM()
        self.sentencetransformer = SentenceTransformerModel()

    def construct(self):
        print("Constructing database graph...")
        print("Adding nodes...")
        self.nodes = [Node(table, llm=self.llm, st=self.sentencetransformer) for table in self.database.get_tables()]
        pdist = 1 - VectorRepresentator(self.database, AttributeValuesDocumentBuilder).get_dist_matrix()
        tables = self.database.get_tables()
        self.graph.add_nodes_from(self.nodes)
        print("Adding edges...")
        for node1 in self.nodes:
            table = node1.table
            for fk in table.get_foreign_keys():
                def get_node(table_name):
                    for n in self.nodes:
                        if n.table.table_name == table_name:
                            return n
                    return None
                edge = Edge(node1, get_node(fk["referred_table"]), self.sentencetransformer)
                self.graph.add_edge(edge.node1, edge.node2)
                self.graph[edge.node1][edge.node2]["edge"] = edge
                self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
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
            else:
                node.page_rank = pr[node]
                node.degree = self.graph.degree(node)
                node.betweenness_centrality = b[node]
                v = node.calculate_feature_vector(self.graph)
                node.embeddings = v["embeddings"].tolist()
                node.features = [node.page_rank, node.degree, node.betweenness_centrality, v["average_semantic_similarity"], v["amount_of_fks"], v["amount_of_columns"], v["row_count"]]
                with open(node_features_file, "wb") as f:
                    pickle.dump({"page_rank": node.page_rank, "degree": node.degree, "betweenness_centrality": node.betweenness_centrality, "embeddings": node.embeddings, "features": node.features}, f)


        print(self.graph.edges)
        print("Database graph constructed.")
        return self.graph


    