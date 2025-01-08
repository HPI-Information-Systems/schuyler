from sqlalchemy import MetaData
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
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
            raise ValueError(f"Error retrieving columns: {e}")

    def get_foreign_keys(self):
        if not self.db.engine:
            print("No active database connection.")
            raise ValueError("No active database connection.")
        try:
            table_name = self.table_name
            fkeys = self.db.inspector.get_foreign_keys(table_name)
            return [{"constrained_columns": fk["constrained_columns"], "referred_table": fk["referred_table"]} for fk in fkeys]
        except SQLAlchemyError as e:
            raise ValueError(f"Error retrieving foreign keys for table '{table_name}': {e}")    
    
    def get_df(self, limit=-1):
        if not self.db.engine:
            print("No active database connection.")
            return None
        try:
            with self.db.engine.connect() as conn:
                query = f"SELECT * FROM {self.table_name}" + f" LIMIT {limit}" if limit > 0 else f"SELECT * FROM {self.table_name}" 
                query = text(query)
                return pd.read_sql(query, conn)
        except SQLAlchemyError as e:
            print(f"Error retrieving rows: {e}")
            return

    def _get_data(self, col, limit=-1):
        try:
            with self.db.engine.connect() as conn:
                query = f"SELECT DISTINCT {col} FROM {self.table_name}" + f" LIMIT {limit}" if limit > 0 else ""
                query = text(query)
                result = conn.execute(query)
                rows = [row[0] for row in result]
                return rows
        except SQLAlchemyError as e:
            raise ValueError(f"Error retrieving data for column '{col}': {e}")
    
    def get_row_count(self):
        if not self.db.engine:
            print("No active database connection.")
            return None
        try:
            with self.db.engine.connect() as conn:
                query = f"SELECT COUNT(*) FROM {self.table_name}"
                query = text(query)
                result = conn.execute(query)
                return result.scalar()
        except SQLAlchemyError as e:
            print(f"Error retrieving row count: {e}")
            

    def _load_table(self):
        try:
            return SQLATable(self.table_name, self.metadata, autoload_with=self.db.engine)
        except SQLAlchemyError as e:
            print(f"Error loading table '{self.table_name}': {e}")
            return None
        
    def get_primary_key(self):
        return self.db.inspector.get_pk_constraint(self.table_name)["constrained_columns"]

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