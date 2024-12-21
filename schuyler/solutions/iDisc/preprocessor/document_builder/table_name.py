from schuyler.solutions.iDisc.preprocessor.document_builder.base_document_builder import BaseDocumentBuilder
from schuyler.database import Database, Table

class TableNameDocumentBuilder(BaseDocumentBuilder):
    def __init__(self):
        super().__init__()
        self.name = "TableName"
    
    def get_document(self, table: Table):
        return table.table_name
    