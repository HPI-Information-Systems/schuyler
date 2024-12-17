import re
from sklearn.feature_extraction.text import TfidfVectorizer

from schuyler.solutions.iDisc.preprocessor.BaseRepresentator import BaseRepresentator
from schuyler.database import Database, Table

class VectorRepresentator(BaseRepresentator):
    def __init__(self, database: Database):
        super().__init__(database)

    def get_representation(self):
        return self.build_tfidf(self.get_documents(), self.tokenize)

    def get_document(self, table: Table):
        cols = table._get_columns()
        return ' '.join(cols)
    
    def build_tfidf(self, documents, tokenizer):
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
        return vectorizer.fit_transform(documents)
    
    def get_documents(self):
        tables = self.database.get_tables()
        return [self.get_document(table) for table in tables]
    
    def tokenize(self, text):
        tokens = re.split(r'[\s_.,;:\-]+', text)
        camel_case_split = []
        for token in tokens:
            camel_case_split.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', token))
        return [token.lower() for token in camel_case_split if token]