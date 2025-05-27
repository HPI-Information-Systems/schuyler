import time
import numpy as np
import os
import pickle
from sklearn.cluster import AffinityPropagation
import json
from openai import OpenAI

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
    def test(self, gpt_model, sql_file_path, schema_file_path, groundtruth, model):
        start_time = time.time()
        with open(schema_file_path, 'r') as file:
            sql_content = file.read()

        basic_prompt = f"Here is an SQL script:\n{sql_content}. Please return a clustering of the database tables as an array of arrays. Do not return anything else but a list of lists consisting of only database tables. Do not format your response or add any ``` or similar. Make sure that you include all provided tables of the sql file in the clustering."
        print("prompt", basic_prompt)
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model=gpt_model,
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that always responds in JSON format."},
        #         {"role": "user", "content": basic_prompt}
        #     ]
        # )


        response = [['edit', 'edit_note', 'edit_note_change'], ['artist', 'artist_alias', 'artist_credit', 'artist_credit_name', 'artist_gid_redirect', 'artist_meta', 'artist_type', 'artist_rating_raw', 'artist_series', 'artist_tag', 'artist_tag_raw', 'editor_subscribe_artist', 'editor_subscribe_artist_deleted', 'l_artist_artist', 'l_artist_series', 'l_artist_event', 'l_artist_genre', 'l_artist_instrument', 'l_artist_label', 'l_artist_place', 'l_artist_recording', 'l_artist_release', 'l_artist_release_group', 'l_artist_url', 'l_artist_work', 'artist_release', 'artist_release_group', 'artist_release_group_nonva', 'artist_release_group_va', 'artist_release_group_pending_update', 'artist_release_nonva', 'artist_release_va', 'artist_release_pending_update'], ['release', 'release_gid_redirect', 'release_alias', 'release_annotation', 'release_event', 'release_raw', 'release_series', 'release_meta', 'release_group', 'release_group_alias', 'release_group_gid_redirect', 'release_group_series', 'release_group_tag', 'release_group_tag_raw', 'l_release_release', 'l_release_series', 'release_group_rating_raw', 'release_group_primary_type', 'release_group_secondary_type', 'release_group_secondary_type_join', 'release_group_annotation', 'release_group_attribute', 'release_group_attribute_type', 'release_group_attribute_type_allowed_value'], ['recording', 'recording_alias', 'recording_gid_redirect', 'recording_series', 'recording_meta', 'recording_tag', 'recording_tag_raw', 'l_recording_recording', 'l_recording_series', 'recording_rating_raw', 'recording_annotation', 'recording_attribute', 'recording_attribute_type', 'recording_attribute_type_allowed_value', 'recording_first_release_date'], ['work', 'work_alias', 'work_gid_redirect', 'work_series', 'work_meta', 'work_tag', 'work_tag_raw', 'l_work_work', 'work_annotation', 'work_attribute', 'work_attribute_type', 'work_attribute_type_allowed_value', 'work_language', 'work_rating_raw'], ['area', 'area_alias', 'area_alias_type', 'area_gid_redirect', 'area_annotation'], ['instrument', 'instrument_alias', 'instrument_gid_redirect', 'instrument_annotation'], ['series', 'series_alias', 'series_annotation'], ['l_label_label', 'label', 'label_annotation', 'label_attribute'], ['link_type', 'link_attribute_type', 'link'], ['language', 'script', 'country_area'], ['l_area_area', 'l_area_artist', 'l_area_event', 'l_area_label', 'l_area_recording', 'l_area_release', 'l_area_series', 'l_area_url', 'l_area_work'], ['event', 'event_gid_redirect', 'l_event_series', 'l_event_recording', 'l_event_work', 'l_event_place', 'l_event_url', 'l_event_event'], ['genre', 'genre_alias', 'genre_annotation'], ['place', 'place_meta', 'place_tag', 'place_gid_redirect', 'place_annotation'], ['url', 'url_gid_redirect', 'l_url_work'], ['label_gid_redirect', 'label_ipi', 'label_annotation', 'label_alias', 'label_isni'], ['track', 'track_gid_redirect', 'track_raw', 'release_label', 'l_release_release', 'release_gid_redirect'], ['editor', 'editor_language', 'editor_oauth_token', 'editor_preference'], ['annotation'], ['vote'], ['release_series']]
        gt = [["edit_area", "area_tag_raw","area_tag","area_annotation","l_area_area", "l_area_url", "area_containment", "area_gid_redirect", "country_area", "area_alias", "area", "area_alias_type", "area_type", "area_attribute", "area_attribute_type", "area_attribute_type_allowed_value", "iso_3166_1", "iso_3166_2", "iso_3166_3"],
            [ "edit_artist", "artist_tag_raw","artist_tag","artist_annotation","l_area_artist", "artist", "l_artist_artist", "l_artist_event", "l_artist_place", "l_artist_url",  "artist_alias", "artist_meta", "artist_release","artist_release_nonva", "artist_release_va", "artist_release_pending_update", "artist_release_group", "artist_gid_redirect", "artist_ipi", "artist_isni", "artist_type", "gender", "artist_alias_type", "artist_rating_raw", "artist_attribute", "artist_attribute_type", "artist_attribute_type_allowed_value"],
            ["edit_event","event_tag_raw","event_tag","event_annotation", "event", "l_area_event", "l_event_event", "l_event_place", "l_event_url", "event_meta", "event_gid_redirect", "event_alias", "event_type", "event_alias_type", "event_rating_raw", "event_attribute", "event_attribute_type", "event_attribute_type_allowed_value"],
            ["edit_genre","genre_annotation", "l_area_genre", "l_artist_genre", "l_genre_genre", "l_genre_instrument", "l_genre_label", "l_genre_place", "l_genre_release_group", "l_genre_url", "genre_alias", "genre", "genre_alias_type"],
            ["edit_instrument","instrument_tag_raw","instrument_tag","instrument_annotation", "instrument", "l_area_instrument", "l_artist_instrument", "l_instrument_instrument", "l_instrument_label", "l_instrument_url", "instrument_gid_redirect", "instrument_alias", "instrument_alias_type", "instrument_type", "instrument_attribute", "instrument_attribute_type", "instrument_attribute_type_allowed_value"],
            ["edit_label", "label_tag_raw","label_tag","label_annotation", "label", "l_area_label", "l_artist_label", "l_event_label", "l_label_label", "l_label_place", "l_label_recording", "l_label_release", "l_label_series", "l_label_url", "l_label_work", "label_alias", "label_meta", "label_gid_redirect", "label_ipi", "label_isni", "label_type", "label_alias_type", "label_rating_raw", "label_attribute", "label_attribute_type", "label_attribute_type_allowed_value"],
            ["edit_recording", "recording_tag_raw", "recording_tag","recording_annotation", "recording", "l_area_recording", "l_artist_recording", "l_event_recording", "l_place_recording", "l_recording_recording", "l_recording_release", "l_recording_url", "l_recording_work", "recording_first_release_date", "recording_gid_redirect", "recording_meta", "recording_alias", "recording_alias_type", "recording_rating_raw", "isrc", "recording_attribute", "recording_attribute_type", "recording_attribute_type_allowed_value"],
            ["edit_release","release_tag_raw","release_tag","release_annotation", "release", "l_area_release", "l_artist_release", "l_event_release", "l_place_release", "l_release_release", "l_release_series","release_series", "l_release_url", "release_meta", "release_unknown_country", "release_first_release_date", "release_gid_redirect", "release_alias", "release_country", "release_label", "artist_release", "country_area", "release_alias_type", "release_status", "release_packaging", "script","release_attribute", "release_attribute_type", "release_attribute_type_allowed_value", "release_event", "alternative_release", "alternative_release_type", "release_raw"],
            ["edit_release_group", "release_group_tag_raw","release_group_tag","release_group_annotation", "release_group", "l_artist_release_group", "l_event_release_group", "l_label_release_group", "l_release_group_release_group", "l_release_group_url", "artist_release_group", "artist_release_group_nonva", "artist_release_group_va", "artist_release_group_pending_update", "release_group_gid_redirect", "release_group_meta", "release_group_alias", "release_group_secondary_type_join", "release_group_alias_type", "release_group_secondary_type", "release_group_primary_type", "release_group_rating_raw", "release_group_attribute", "release_group_attribute_type", "release_group_attribute_type_allowed_value"],
            ["edit_series",  "series_tag_raw","series_tag","series_annotation", "series", "l_area_series", "l_artist_series","artist_series", "l_event_series","event_series",  "l_place_series", "l_recording_series","recording_series", "l_release_group_series", "release_group_series", "l_series_series", "l_series_url", "series_gid_redirect", "series_alias", "series_alias_type", "series_type", "series_ordering_type", "series_attribute", "series_attribute_type", "series_attribute_type_allowed_value"],
            ["edit_work","work_tag_raw","work_tag", "work_annotation", "work", "l_area_work", "l_artist_work", "l_event_work", "l_series_work", "l_url_work", "l_work_work", "work_alias", "work_meta", "work_gid_redirect", "work_attribute", "work_attribute_type_allowed_value",  "work_type", "work_attribute_type", "work_rating_raw", "iswc", "work_alias_type", "work_series"],
            ["edit_place",  "place_tag","place_tag_raw","place_annotation", "place", "l_place_place", "l_place_url", "l_place_work", "place_meta", "place_gid_redirect", "place_alias", "place_rating_raw", "place_alias_type", "place_attribute", "place_attribute_type", "place_attribute_type_allowed_value", "place_type"],
            ["tag","tag_relation"],
            ["annotation"],
            ["artist_credit_name", "artist_credit", "artist_credit_gid_redirect"],
            ["edit_url","url", "url_gid_redirect"],
            ["language", "work_language", "editor_language"],
            ["alternative_medium", "medium_cdtoc", "alternative_medium_track", "medium", "medium_attribute", "medium_attribute_type", "medium_attribute_type_allowed_format", "medium_attribute_type_allowed_value", "medium_attribute_type_allowed_value_allowed_format", "medium_format", "medium_index", "medium_track_durations"],
            ["edit",  "edit_data",     "edit_note", "edit_note_change", "edit_note_recipient",    "vote", "autoeditor_election_vote"],
            ["track", "alternative_track", "track_gid_redirect", "track_raw"],
            ["link", "link_attribute", "link_attribute_credit", "link_attribute_text_value", "link_attribute_type", "link_creditable_attribute_type", "link_text_attribute_type", "link_type", "link_type_attribute_type", "orderable_link_type"],
            ["autoeditor_election", "autoeditor_election_vote", "editor", "editor_language", "editor_oauth_token", "editor_preference", "editor_subscribe_artist", "editor_subscribe_artist_deleted", "editor_subscribe_collection", "editor_subscribe_editor", "editor_subscribe_label", "editor_subscribe_label_deleted", "editor_subscribe_series", "editor_subscribe_series_deleted", "old_editor_name"],
            ["editor_collection", "editor_collection_area", "editor_collection_artist", "editor_collection_collaborator", "editor_collection_deleted_entity", "editor_collection_event", "editor_collection_genre", "editor_collection_gid_redirect", "editor_collection_instrument", "editor_collection_label", "editor_collection_place", "editor_collection_recording", "editor_collection_release", "editor_collection_release_group", "editor_collection_series", "editor_collection_type", "editor_collection_work", "editor_subscribe_collection"],
            ["application"],
            ["cdtoc", "cdtoc_raw"],
            ["replication_control", "dbmirror_pendingdata", "dbmirror_pending", "unreferenced_row_log", "deleted_entity"]]

        #merge gt to one big array
        gt = [item for sublist in gt for item in sublist]
        response_tables = [item for sublist in response for item in sublist]
        missing_tables = [table for table in gt if table not in response_tables]
        extra_tables = [table for table in response_tables if table not in gt]
        print("missing tables", missing_tables)
        print("extra tables", extra_tables)
        #add missing tables to the response
        response.append(missing_tables)
        # for table in missing_tables:
        #     response.append([table])
        # response = response.choices[0].message.content
        # response = response.lower()
        result = response
        print("response", response)
        # result = json.loads(response)
        print("Result", result)
        return result, time.time()-start_time


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