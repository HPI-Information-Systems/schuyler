from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import os
import subprocess
import wandb

from schuyler.database.table import Table

class Database:
    """
    A utility class to handle PostgreSQL connections and metadata operations using SQLAlchemy.
    """

    def __init__(self, username, password, host, port, database):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.engine = None
        self.inspector = None
        self.connect()

    def connect(self):
        try:
            connection_url = (
                f"postgresql+psycopg2://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            self.engine = create_engine(connection_url)
            self.inspector = inspect(self.engine)
            print("Database connection established successfully.")
        except SQLAlchemyError as e:
            print(f"Error connecting to the database: {e}")
            self.engine = None

    def get_tables(self):
        if not self.engine:
            print("No active database connection.")
            return []
        table_names = self.inspector.get_table_names()
        return [Table(self, table_name) for table_name in table_names]

    def get_columns(self, table_name):
        if not self.engine:
            print("No active database connection.")
            return []
        try:
            columns = self.inspector.get_columns(table_name)
            return [{"name": col["name"], "type": str(col["type"])} for col in columns]
        except SQLAlchemyError as e:
            print(f"Error retrieving columns for table '{table_name}': {e}")
            return []
        
    def update_database(self, script_path):
        """Runs an SQL script file using psql command line."""
        wandb.save(script_path, base_path=os.path.dirname(script_path))
        command = ['psql', '-f', script_path, '-U', self.user, '-d', 'postgres', '-h', self.host, '-v', 'ON_ERROR_STOP=1']
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