import time

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution

from schuyler.solutions.iDisc.base_clusterer import SimilarityBasedClusterer

class iDiscSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self, representators, meta_clusterer, sim_clust, link_clust):
        return self.test(representators, meta_clusterer, sim_clust, link_clust)

    def train(self):
        print("No training process required for iDisc.")
        return None, None

    def test(self, representators, meta_clusterer, sim_clust, link_clust, model=None):
        start_time = time.time()
        self.vector_representators = [vr['module'](self.database, **vr["params"]) for vr in representators["vector"]] if "vector" in representators else []
        self.similarity_representators = [sr['module'](self.database,**sr["params"]) for sr in representators["similarity"]] if "similarity" in representators else []
        self.graph_representators = [gr['module'](self.database, **gr["params"]) for gr in representators["graph"]] if "graph" in representators else []
        #todo the meta clusterer consists of multiple clusterers
        #self.meta_clusterer = meta_clusterer["module"](**meta_clusterer["params"])
        self.sim_clust = SimilarityBasedClusterer(**sim_clust)
        #self.link_clust = link_clust["module"](**link_clust)

        representations = []
        # for vr in self.vector_representators:
        #     representations.append(vr.get_representation())
        # for sr in self.similarity_representators:
        #     representations.append(sr.get_representation())
        print("representations", representations)
        s = self.similarity_representators[0]
        tables = list(map(lambda t: t.table_name, self.database.get_tables()))
        self.sim_clust.cluster(s, tables)
        clusters = []
        #output["metrics"], output["runtime"]
        return clusters, time.time()-start_time


        
