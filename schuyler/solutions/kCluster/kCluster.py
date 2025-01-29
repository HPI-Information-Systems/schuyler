import time

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.kCluster.table_importance import TableImportance

class kClusterSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__
        print("NOT IMPLEMENTED: CHECK THAT TWICE THE SAME CLUSTERER / REPRESENTATOR IS NOT USED")

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for iDisc.")
        return None, None

    def test(self, groundtruth, model=None):
        start_time = time.time()
        #calculate Taböle importance
        ti = TableImportance(self.database)


        clusters = []
        return clusters, time.time()-start_time


        
