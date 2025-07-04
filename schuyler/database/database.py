from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import os
import subprocess
from sqlalchemy import text
import wandb

from schuyler.database.table import Table

class Database:
    def __init__(self, username, password, host, port, database, schema=None):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.schema = schema
        self.engine = None
        self.inspector = None
        self.connect()

    def connect(self):
        try:
            connection_url = (
                f"postgresql+psycopg2://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            if self.schema:
                connection_url += f"?options=-csearch_path={self.schema}"
            print("SHC",connection_url)
            self.engine = create_engine(connection_url) #if not self.schema else create_engine(connection_url, connect_args={'options': '-csearch_path={}'.format(self.schema)})
            self.inspector = inspect(self.engine)

            
            print("Database connection established successfully.")
        except SQLAlchemyError as e:
            self.engine = None
            raise ValueError(f"Error connecting to database: {e}")
    
    def get_tables(self):
        """
        Retrieves all tables from the database and returns a list of Table objects.
        """
        if not self.engine:
            raise ValueError("No active database connection.")
        table_names = sorted(self.inspector.get_table_names())
        return [Table(self, table_name) for table_name in table_names]
    
    def get_foreign_keys(self):
        if not self.engine:
            raise ValueError("No active database connection.")
        try:
            fks = []
            for table in self.get_tables():
                fks.extend(table.get_foreign_keys())
            return fks
        except SQLAlchemyError as e:
            raise ValueError(f"Error retrieving foreign keys: {e}")

    # def is_one_to_one(self, table_name):


    def get_columns(self, table_name):
        if not self.engine:
            print("No active database connection.")
            return []
        try:
            columns = self.inspector.get_columns(table_name)
            return [{"name": col["name"], "type": str(col["type"])} for col in columns]
        except SQLAlchemyError as e:
            raise ValueError(f"Error retrieving columns for table '{table_name}': {e}")

    def execute_query(self, query):
        if not self.engine:
            raise ValueError("No active database connection.")
        with self.engine.connect() as connection:
            query = text(query)
            result = connection.execute(query)
            return result.fetchall()

    @staticmethod 
    def update_database(script_path):
        """Runs an SQL script file using psql command line."""
        #wandb.save(script_path, base_path=os.path.dirname(script_path))
        command = ['psql', '-f', script_path, '-U', os.getenv("POSTGRES_USER"), '-d', 'postgres', '-h', os.getenv("POSTGRES_HOST"), '-v', 'ON_ERROR_STOP=1']
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            print("Database updated successfully.")
        else:
            print("Database update failed with return code:", result.returncode)
            print("Error output:", result.stderr)
            raise ValueError("Database update failed")

    def close(self):
        if self.engine:
            self.engine.dispose()
            print("Database connection closed.")