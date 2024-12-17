from sqlalchemy import MetaData
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text, Table as SQLATable
class Table:
    def __init__(self, db, table_name):
        self.db = db
        self.table_name = table_name
        self.metadata = MetaData()
        self.columns = self._get_columns()
        self.table = self._load_table()

    def _get_columns(self):
        try:
            return self.db.inspector.get_columns(self.table_name)
        except SQLAlchemyError as e:
            print(f"Error retrieving columns: {e}")
            return []

    def _load_table(self):
        try:
            return SQLATable(self.table_name, self.metadata, autoload_with=self.db.engine)
        except SQLAlchemyError as e:
            print(f"Error loading table '{self.table_name}': {e}")
            return None

    def get_all_rows(self):
        if not self.db.engine:
            print("No active database connection.")
            return []
        try:
            with self.db.engine.connect() as conn:
                query = text(f"SELECT * FROM {self.table_name}")
                result = conn.execute(query)
                rows = [dict(row) for row in result]
                return rows
        except SQLAlchemyError as e:
            print(f"Error retrieving rows: {e}")
            return []

    def get_values_for_attribute(self, column_name):
        if not self.db.engine:
            print("No active database connection.")
            return []
        try:
            with self.db.engine.connect() as conn:
                query = text(f"SELECT DISTINCT {column_name} FROM {self.table_name}")
                result = conn.execute(query)
                values = [row[0] for row in result]
                return values
        except SQLAlchemyError as e:
            print(f"Error retrieving values for column '{column_name}': {e}")
            return []

    def __str__(self):
        column_info = ", ".join([f"{col['name']} ({col['type']})" for col in self.columns])
        return f"Table '{self.table_name}' with columns: {column_info}"