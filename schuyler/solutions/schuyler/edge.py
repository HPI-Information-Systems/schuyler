from sentence_transformers import util
from schuyler.solutions.schuyler.node import Node
import numpy as np
# - edge features, such as
# - similarity of column names via LLMs
# - semantische domäne -> Ähmlichkeit der semantische domöne 

# - overlap of column names
# - overlap of attribute sets
# - fraction of foreign keys
# - cardinality of foreign keys
# - edge betweenness centrality
# - somehow exploit attention mechanism

class Edge:
    def __init__(self, node1: Node, node2: Node, st, sim=None):
        self.node1 = node1
        self.node2 = node2
        self.st = st
        self.weight = None
        if sim == None:
            self.table_sim = self.get_table_similarity()
        else:
            self.table_sim = sim
        print(f"Table {node1.table.table_name} and {node2.table.table_name} have a similarity of {self.table_sim}")

    def normalize(self, vector):
        if vector.is_cuda:
            vector = vector.cpu()
        vector_np = vector.numpy()
        return vector_np / np.linalg.norm(vector_np)

    def set_weight_attr_to_attr(self, attr):
        self.__setattr__("weight", self.__getattribute__(attr))

    def get_table_similarity(self):
        em1 = self.st.encode(self.node1.llm_description)
        em2 = self.st.encode(self.node2.llm_description)
        return util.cos_sim(em1, em2).item()
    




# def semantic_similarity(col1: str, col2: str) -> float:
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embedding1 = model.encode(col1)
#     embedding2 = model.encode(col2)
#     return cosine_similarity([embedding1], [embedding2])[0][0]

# def semantic_table_similarity(table1: Table, table2: Table) -> float:
#     columns_1 = [col["name"] for col in table1.columns]
#     columns_2 = [col["name"] for col in table2.columns]
#     def build_prompt
#     similarities = [semantic_similarity(col1, col2) for col1 in columns_1 for col2 in columns_2]
#     return sum(similarities) / len(similarities)



