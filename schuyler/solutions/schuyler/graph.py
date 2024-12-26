from schuyler.database.database import Database
from networkx import Graph

class DatabaseGraph:
    def __init__(self, database: Database):
        self.graph = Graph()
        self.database = database

    def construct(self):
        self.nodes = self.database.get_tables()
        self.graph.add_nodes_from(self.nodes)
        for table in self.nodes:
            for fk in self.database.get_foreign_keys(table):
                self.graph.add_edge(table, fk["referred_table"])
        return self.graph

    