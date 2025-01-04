from schuyler.solutions.iDisc.preprocessor.document_builder.base_document_builder import BaseDocumentBuilder
from schuyler.database import Database, Table

class AttributeValuesDocumentBuilder(BaseDocumentBuilder):
    def __init__(self):
        super().__init__()
        self.name = "AttributeValuesDocumentBuilder"
    
    def get_document(self, table: Table):
        cols = table._get_columns()
        meta = []
        for c in cols:
            data = table._get_data(c['name'], 40)
            #convert to string #
            data = [str(d) for d in data]
            meta.append(c['name'] + " " + ' '.join(data))
        return table.table_name + " " + ' '.join(meta)
    