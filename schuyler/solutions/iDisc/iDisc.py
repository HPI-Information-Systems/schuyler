from sklearn.base import BaseEstimator, ClusterMixin

from schuyler.database.database import Database

class iDisc(BaseEstimator, ClusterMixin):
    def __init__(self, database: Database):
        self.database = database

    def train(self):
        print("No training process required for iDisc.")

    def test(self, representators, meta_clusterer, sim_clust, link_clust):

        self.vector_representators = [vr['module'](**vr["params"],self.database) for vr in representators["vector"]]
        self.similarity_representators = [sr['module'](**sr["params"],self.database) for sr in representators["similarity"]]
        self.graph_representators = [gr['module'](**gr["params"],self.database) for gr in representators["graph"]]
        self.meta_clusterer = meta_clusterer["module"](**meta_clusterer["params"])
        self.sim_clust = sim_clust["module"](**sim_clust["params"])
        self.link_clust = link_clust["module"](**link_clust["params"])
        
