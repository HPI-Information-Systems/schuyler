import os
import subprocess
import pandas as pd
import wandb
from dotenv import load_dotenv
from sqlalchemy import create_engine

class PostgresClient:
    def __init__(self, database):
        load_dotenv()
        self.host = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.database = database
        #self.engine = self.get_database_engine()

    def get_database_engine(self):
        """Creates and returns a SQLAlchemy engine for connecting to the database."""
        try:
            engine_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            print("Connecting to database:", engine_url)
            engine = create_engine(engine_url)
            return engine
        except Exception as e:
            print(f"Error creating SQLAlchemy engine: {e}")
            return None

    def update_database(self, script_path, add_primary_keys=False):
        """Runs an SQL script file using psql command line."""
        wandb.save(script_path, base_path=os.path.dirname(script_path))
        command = ['psql', '-f', script_path, '-U', self.user, '-d', 'postgres', '-h', self.host, '-v', 'ON_ERROR_STOP=1']
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        # catch error if update failed
        
        if result.returncode == 0:
            print("Database updated successfully.")
        else:
            print("Database update failed with return code:", result.returncode)
            print("Error output:", result.stderr)
            raise ValueError("Database update failed")
        if add_primary_keys:
            self.add_primary_keys()

    

    def get_all_tables(self):
        """Retrieves all tables from the database and returns a dictionary with DataFrames for each table."""
        tables = {}
        try:
            engine = self.get_database_engine()
            # Get the list of tables
            table_names = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", engine)
            table_names = table_names['table_name'].tolist()

            # Load each table into a DataFrame
            for table_name in table_names:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql(query, engine)
                tables[table_name] = df
                print(f"Table '{table_name}' loaded successfully.")

            return tables

        except Exception as e:
            print(f"Error fetching tables: {e}")
            return {}

