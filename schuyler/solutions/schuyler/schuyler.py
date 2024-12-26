import time
import networkx as nx

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph

class SchuylerSolution(BaseSolution):
    def __init__(self, database: Database):
        self.database = database
        self.solution_name = self.__class__.__name__

    def run(self):
        return self.test()

    def train(self):
        print("No training process required for Schuyler.")
        return None, None

    def test(self, model):
        start_time = time.time()
        G = DatabaseGraph(self.database)
        G.construct()
        clusters = nx.community.louvain_communities(G.graph, weight="weight")
        print(clusters)
        for cluster in clusters:
            print(cluster)
        
        #{holding_summary, account_permission, customer_account, holding}, {customer, watch_item, taxrate, customer_taxrate, watch_list}, {commission_rate, zip_code, exchange, address}, {trade_request, broker, trade_type, cash_transaction, trade, charge, status_type, holding_history, settlement, trade_history}, {daily_market, last_trade, security}, {industry, sector, financial, news_item, news_xref, company_competitor, company}
        #convert list as defined above into list of lists
        result = []
        for cluster in clusters:
            result.append([str(table) for table in cluster])
        return result, time.time()-start_time


        
