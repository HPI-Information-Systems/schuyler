import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist,squareform

from schuyler.solutions.iDisc.preprocessor.BaseRepresentator import BaseRepresentator
from schuyler.database import Database, Table
from schuyler.solutions.utils import tokenize

class VectorRepresentator(BaseRepresentator):
    def __init__(self, database: Database, document_builder, vectorizer='tfidf'):
        super().__init__(database)
        self.document_builder = document_builder()
        self.vectorizer = vectorizer
        self.name = "VectorRepresentator"

    def get_representation(self):
        return self.build_tfidf(self.document_builder.get_documents(self.database), tokenize)
    
    def get_dist_matrix(self):
        sim = self.get_representation().toarray()
        return pdist(sim, metric='cosine')
        
    def build_tfidf(self, documents, tokenizer):
        print("Building TF-IDF matrix")
        print(", ".join(f"Idx{index}: {value}" for index, value in enumerate(documents)))
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
        tra = vectorizer.fit_transform(documents)
        print("TF-IDF matrix built")
        return tra