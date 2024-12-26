from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from schuyler.database.table import Table
from schuyler.solutions.utils import tokenize
# - edge features, such as
# - similarity of column names via LLMs
# - semantische domäne -> Ähmlichkeit der semantische domöne 

# - overlap of column names
# - overlap of attribute sets
# - fraction of foreign keys
# - cardinality of foreign keys
# - edge betweenness centrality
# - somehow exploit attention mechanism

def table_name_semantic_similarity(table1: Table, table2: Table, llm_model="all-MiniLM-L6-v2") -> float:
    model = SentenceTransformer(llm_model)
    embedding1 = model.encode(table1.table_name)
    embedding2 = model.encode(table2.table_name)
    return cosine_similarity([embedding1], [embedding2])[0][0]

def create_table_description(table: Table, llm="") -> str:



def jaccard_of_columnnames(table_1: Table, table_2: Table) -> float:
    columnnames_1 = [col["name"] for col in table_1.columns]
    columnnames_2 = [col["name"] for col in table_2.columns]
    return len(set(columnnames_1).intersection(columnnames_2)) / len(set(columnnames_1).union(columnnames_2))



def semantic_similarity(col1: str, col2: str) -> float:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding1 = model.encode(col1)
    embedding2 = model.encode(col2)
    return cosine_similarity([embedding1], [embedding2])[0][0]

def semantic_table_similarity(table1: Table, table2: Table) -> float:
    columns_1 = [col["name"] for col in table1.columns]
    columns_2 = [col["name"] for col in table2.columns]
    def build_prompt
    similarities = [semantic_similarity(col1, col2) for col1 in columns_1 for col2 in columns_2]
    return sum(similarities) / len(similarities)



