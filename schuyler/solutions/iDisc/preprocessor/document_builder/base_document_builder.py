class BaseDocumentBuilder:
    def get_documents(self, database):
        tables = database.get_tables()
        return [self.get_document(table) for table in tables]
    