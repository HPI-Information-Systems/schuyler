from schuyler.solutions.iDisc.preprocessor.document_builder.base_document_builder import BaseDocumentBuilder
from schuyler.database import Database, Table

class TableNameAndColsDocumentBuilder(BaseDocumentBuilder):
    def __init__(self):
        super().__init__()
        self.name = "TableNameAndCols"
    
    def get_document(self, table: Table):
        cols = table._get_columns()
        return table.table_name + " " + ' '.join(list(map(lambda c: c['name'], cols)))
    