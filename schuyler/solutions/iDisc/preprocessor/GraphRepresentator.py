import networkx as nx

from schuyler.solutions.iDisc.preprocessor.BaseRepresentator import BaseRepresentator

class GraphRepresentator(BaseRepresentator):
    def __init__(self, database):
        super().__init__(database)
        self.name = "GraphRepresentator"

    def get_representation(self):
        tables = self.database.get_tables()
        G = nx.Graph()
        for table in tables:
            G.add_node(table.table_name)
            fks = list(map(lambda fk: (table.table_name, fk["referred_table"]), table.get_foreign_keys()))
            G.add_edges_from(fks)
        return G


