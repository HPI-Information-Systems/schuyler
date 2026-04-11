from abc import ABC, abstractmethod
from schuyler.database.database import Database

class BaseSolution(ABC):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def train(self):
        raise NotImplementedError(f"{self.solution_name}.train is not implemented")
    
    def predict(self):
        raise NotImplementedError(f"{self.solution_name}.predict is not implemented")
