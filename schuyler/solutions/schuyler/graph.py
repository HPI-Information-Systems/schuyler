from schuyler.database.database import Database
from networkx import Graph, pagerank, betweenness_centrality
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.feature_vector.llm import LLM, SentenceTransformerModel
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
        #build dataframe
        self.graph.add_nodes_from(self.nodes)
        print("Adding edges...")
        for node1 in self.nodes:
            table = node1.table
            # for node2 in self.nodes:
            #     edge = Edge(node1, node2, self.sentencetransformer)
            #     self.graph.add_edge(edge.node1, edge.node2)
            #     self.graph[edge.node1][edge.node2]["edge"] = edge
            #     self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
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
        pr = pagerank(self.graph, alpha=0.85)
        b = betweenness_centrality(self.graph, normalized=True, endpoints=False, seed=42)
        for node in self.nodes:
            node.page_rank = pr[node]
            node.degree = self.graph.degree(node)
            node.betweenness_centrality = b[node]
            v = node.calculate_feature_vector(self.graph)
            node.embeddings = v["embeddings"].tolist()
            node.features = [node.page_rank, node.degree, node.betweenness_centrality, v["average_semantic_similarity"], v["amount_of_fks"], v["amount_of_columns"], v["row_count"]]



        print("Database graph constructed.")
        return self.graph

    