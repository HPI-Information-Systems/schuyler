from schuyler.database.database import Database
from networkx import Graph
from schuyler.solutions.schuyler.node import Node
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.feature_vector.llm import LLM, SentenceTransformerModel
class DatabaseGraph:
    def __init__(self, database: Database):
        self.graph = Graph()
        self.database = database
        self.llm = LLM()
        self.sentencetransformer = SentenceTransformerModel()

    def construct(self):
        print("Constructing database graph...")
        print("Adding nodes...")
        self.nodes = [Node(table, llm=self.llm) for table in self.database.get_tables()]
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
                # self.graph[edge.node1][edge.node2]["weight"] = edge.table_sim
        print("Database graph constructed.")
        return self.graph

    