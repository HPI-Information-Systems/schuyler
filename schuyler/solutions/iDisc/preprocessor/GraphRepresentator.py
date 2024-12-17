from schuyler.solutions.iDisc.preprocessor.BaseRepresentator import BaseRepresentator

class GraphRepresentator(BaseRepresentator):
    def __init__(self, database):
        super().__init__(database)

    def get_representation(self, data):
        raise NotImplementedError("GraphRepresentator.get_representation is not implemented")
