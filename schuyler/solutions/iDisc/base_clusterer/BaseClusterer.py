from abc import ABC, abstractmethod

class BaseClusterer(ABC):
    @abstractmethod
    def cluster(self, data):
        pass

    def __str__(self):
        return self.__class__.__name__
