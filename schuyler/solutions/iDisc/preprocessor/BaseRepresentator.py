from abc import ABC, abstractmethod

from schuyler.database.database import Database

class BaseRepresentator(ABC):
    def __init__(self, database: Database):
        self.database = database

    @abstractmethod
    def get_representation(self, data):
        pass

    def __str__(self):
        return self.__class__.__name__