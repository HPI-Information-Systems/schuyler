import time
import numpy as np
import os
import pickle
from sklearn.cluster import AffinityPropagation
import json

from schuyler.database.database import Database
from schuyler.solutions.base_solution import BaseSolution
from schuyler.solutions.schuyler.graph import DatabaseGraph
from schuyler.solutions.schuyler.meta_clusterer import MetaClusterer
from schuyler.solutions.schuyler.clusterer import louvain_clustering, affinity_propagation_clustering,leiden_clustering,affinity_propagation_clustering_with_pca

class GPTSolution(BaseSolution):
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
        #/home/lukas/schuyler/data/output.jsonl

        #magento_cluster = [["admin_assert", "admin_role", "admin_rule", "admin_user", "adminnotification_inbox"], ["amazonpayments_api_debug", "chronopay_api_debug", "cybermut_api_debug", "cybersource_api_debug", "eway_api_debug", "flo2cash_api_debug", "googlecheckout_api_debug", "ideal_api_debug", "paybox_api_debug", "paygate_authorizenet_debug", "paypal_api_debug", "paypaluk_api_debug", "protx_api_debug"], ["api_assert", "api_role", "api_rule", "api_session", "api_user"], ["catalog_category_entity", "catalog_category_entity_datetime", "catalog_category_entity_decimal", "catalog_category_entity_int", "catalog_category_entity_text", "catalog_category_entity_varchar", "catalog_category_flat", "catalog_category_product", "catalog_category_product_index"], ["catalog_compare_item"], ["catalog_product_bundle_option", "catalog_product_bundle_option_value", "catalog_product_bundle_price_index", "catalog_product_bundle_selection"], ["catalog_product_enabled_index", "catalog_product_entity", "catalog_product_entity_datetime", "catalog_product_entity_decimal", "catalog_product_entity_gallery", "catalog_product_entity_int", "catalog_product_entity_media_gallery", "catalog_product_entity_media_gallery_value", "catalog_product_entity_text", "catalog_product_entity_tier_price", "catalog_product_entity_varchar", "catalog_product_link", "catalog_product_link_attribute", "catalog_product_link_attribute_decimal", "catalog_product_link_attribute_int", "catalog_product_link_attribute_varchar", "catalog_product_link_type", "catalog_product_option", "catalog_product_option_price", "catalog_product_option_title", "catalog_product_option_type_price", "catalog_product_option_type_title", "catalog_product_option_type_value", "catalog_product_super_attribute", "catalog_product_super_attribute_label", "catalog_product_super_attribute_pricing", "catalog_product_super_link", "catalog_product_website"], ["catalogindex_aggregation", "catalogindex_aggregation_tag", "catalogindex_aggregation_to_tag", "catalogindex_eav", "catalogindex_minimal_price", "catalogindex_price"], ["cataloginventory_stock", "cataloginventory_stock_item", "cataloginventory_stock_status"], ["catalogrule", "catalogrule_affected_product", "catalogrule_product", "catalogrule_product_price"], ["catalogsearch_fulltext", "catalogsearch_query", "catalogsearch_result"], ["checkout_agreement", "checkout_agreement_store"], ["cms_block", "cms_block_store", "cms_page", "cms_page_store"], ["core_config_data", "core_email_template", "core_flag", "core_layout_link", "core_layout_update", "core_resource", "core_session", "core_store", "core_store_group", "core_translate", "core_url_rewrite", "core_website", "cron_schedule"], ["customer_address_entity", "customer_address_entity_datetime", "customer_address_entity_decimal", "customer_address_entity_int", "customer_address_entity_text", "customer_address_entity_varchar", "customer_entity", "customer_entity_datetime", "customer_entity_decimal", "customer_entity_int", "customer_entity_text", "customer_entity_varchar", "customer_group"], ["dataflow_batch", "dataflow_batch_export", "dataflow_batch_import", "dataflow_import_data", "dataflow_profile", "dataflow_profile_history", "dataflow_session"], ["design_change"], ["directory_country", "directory_country_format", "directory_country_region", "directory_country_region_name", "directory_currency_rate"], ["downloadable_link", "downloadable_link_price", "downloadable_link_purchased", "downloadable_link_purchased_item", "downloadable_link_title", "downloadable_sample", "downloadable_sample_title"], ["eav_attribute", "eav_attribute_group", "eav_attribute_option", "eav_attribute_option_value", "eav_attribute_set", "eav_entity", "eav_entity_attribute", "eav_entity_datetime", "eav_entity_decimal", "eav_entity_int", "eav_entity_store", "eav_entity_text", "eav_entity_type", "eav_entity_varchar"], ["gift_message"], ["googlebase_attributes", "googlebase_items", "googlebase_types", "googleoptimizer_code"], ["newsletter_problem", "newsletter_queue", "newsletter_queue_link", "newsletter_queue_store_link", "newsletter_subscriber", "newsletter_template"], ["poll", "poll_answer", "poll_store", "poll_vote"], ["product_alert_price", "product_alert_stock"], ["rating", "rating_entity", "rating_option", "rating_option_vote", "rating_option_vote_aggregated", "rating_store", "rating_title"], ["review", "review_detail", "review_entity", "review_entity_summary", "review_status", "review_store"], ["sales_flat_order_item", "sales_flat_quote", "sales_flat_quote_address", "sales_flat_quote_address_item", "sales_flat_quote_item", "sales_flat_quote_item_option", "sales_flat_quote_payment", "sales_flat_quote_shipping_rate", "sales_order", "sales_order_datetime", "sales_order_decimal", "sales_order_entity", "sales_order_entity_datetime", "sales_order_entity_decimal", "sales_order_entity_int", "sales_order_entity_text", "sales_order_entity_varchar", "sales_order_int", "sales_order_tax", "sales_order_text", "sales_order_varchar"], ["salesrule", "salesrule_customer"], ["shipping_tablerate"], ["sitemap"], ["tag", "tag_relation", "tag_summary"], ["tax_calculation", "tax_calculation_rate", "tax_calculation_rate_title", "tax_calculation_rule", "tax_class"], ["weee_discount", "weee_tax"], ["wishlist", "wishlist_item"]]
        tpcee_cluster = [["account_permission", "customer", "customer_account", "customer_taxrate", "address", "zip_code"], ["broker", "commission_rate", "trade", "trade_history", "trade_request", "trade_type", "settlement", "cash_transaction", "charge"], ["company", "company_competitor", "industry", "sector"], ["daily_market", "exchange", "financial", "holding", "holding_history", "holding_summary", "last_trade", "security"], ["news_item", "news_xref"], ["status_type", "taxrate"], ["watch_item", "watch_list"]]
        
        
        return tpcee_cluster, time.time()-start_time
    


    # def test(self, model):
    #     start_time = time.time()
    #     tables = self.database.get_tables()
    #     # fk_constraints = self.database.get_foreign_keys()

    #     prompt = "Cluster the tables in the database into topically coherent clusters. The output should be a list of lists including only the table names with each list representing one cluster. Do not output anything else! Do not output anything in the beginning or in the end, just output the clustering. Return this list of lists as plain text and not as json or so. Return everything in one line. The tables are: "
    #     for table in tables:
    #         prompt += table.table_name + ", "
    #     prompt = prompt[:-2] + "."
    #     #prompt = prompt[:-2] + ". The foreign key constraints are: "
    #     # for fk in fk_constraints:
    #     #     prompt += fk.table_name + " references " + fk.referenced_table_name + ", "
    #     # prompt = prompt[:-2] + "."
    #     database_name = self.database.database
    #     prompt_structure = lambda prompt: {
    #         "model": "gpt-4o",
    #         "messages": [
    #             {"role": "system", "content": "You are a helpful assistant who is specialized in clustering databases."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         "max_tokens": 15000,
    #         "temperature": 0.7,
    #         "top_p": 1.0,
    #         "frequency_penalty": 0.0,
    #         "presence_penalty": 0.0
    #     }
    #     prompt_file = f"/data/{database_name}_prompt.jsonl"
    #     os.makedirs(f"/data", exist_ok=True)
    #     with open(prompt_file, "w") as f:
    #         f.write(json.dumps(prompt_structure(prompt)))
    #     return None, time.time()-start_time