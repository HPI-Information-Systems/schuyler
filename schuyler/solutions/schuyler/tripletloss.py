import networkx as nx
import random
import torch
from sentence_transformers import InputExample
from schuyler.solutions.schuyler.edge import Edge
from schuyler.solutions.schuyler.node import Node

G = nx.Graph()

# Precompute all node pairs for negative sampling
all_nodes = list(G.nodes())
node_set = set(all_nodes)

num_hard_negatives = 1  # Number of hard negatives per anchor-positive pair
similarity_threshold = 0.5  # Minimum similarity score for a hard negative
top_k = 5  # Number of top similar negatives to consider

def generate_triplets(
    G: nx.Graph,
    st,
    num_triplets_per_anchor: int = 5,
    similarity_threshold: float = 0.5,
):
    triplets = []
    all_nodes = list(G.nodes())
    print(all_nodes)
    print(G.nodes())
    for anchor in G.nodes():
        print("Anchor: ", anchor)
        positives = list(G.neighbors(anchor))
        print("Positives: ", positives)
        if not positives:
            continue
        
        for _ in range(num_triplets_per_anchor):
            positive = random.choice(positives)
            potential_negatives = [n for n in all_nodes if n != anchor and n not in G.neighbors(anchor)]
            
            if not potential_negatives:
                continue
            # filtered_negatives = [
            #     neg for neg in potential_negatives
            #     if compute_similarity(anchor, neg) < similarity_threshold
            # ]
            filtered_negatives = []
            # for neg in potential_negatives:
            #     sim = Edge(anchor, neg, st).get_table_similarity()
            #     if sim < similarity_threshold:
            #         filtered_negatives.append(neg)
            
            # if not filtered_negatives:
            #     continue
            negative = random.choice(potential_negatives)
            print("Negative: ", negative)
            triplets.append((anchor, positive, negative))
    return triplets

def generate_similar_and_nonsimilar_triplets(
    G: nx.Graph,
    sim_matrix,
    num_triplets_per_anchor: int = 5,
    high_similarity_threshold = 0.6,
    low_similarity_threshold = 0.25,
):
    triplets = []
    all_nodes = list(G.nodes())
    print(all_nodes)
    print(G.nodes())
    for anchor in G.nodes():
        print("Anchor: ", anchor)
        positives = list(G.neighbors(anchor))
        #only select very similar positives#
        positives = [pos for pos in positives if sim_matrix.loc[str(anchor), str(pos)] > high_similarity_threshold]
        print("Positives: ", positives)
        if not positives:
            continue
        
        for _ in range(num_triplets_per_anchor):
            positive = random.choice(positives)
            potential_negatives = [n for n in all_nodes if n != anchor and n not in G.neighbors(anchor)]
            
            if not potential_negatives:
                continue
            # filtered_negatives = [
            #     neg for neg in potential_negatives
            #     if compute_similarity(anchor, neg) < similarity_threshold
            # ]
            filtered_negatives = []
            # only select very dissimilar negatives
            potential_negatives = [neg for neg in potential_negatives if sim_matrix.loc[str(anchor), str(neg)] < low_similarity_threshold]
            # for neg in potential_negatives:
            #     sim = Edge(anchor, neg, st).get_table_similarity()
            #     if sim < similarity_threshold:
            #         filtered_negatives.append(neg)
            
            # if not filtered_negatives:
            #     continue
            if not potential_negatives:
                continue
            negative = random.choice(potential_negatives)
            print("Negative: ", negative)
            triplets.append((anchor, positive, negative))
    return triplets


def generate_hard_triplets(G, node_descriptions, node_embeddings, st, num_triplets_per_anchor=5, top_k=5, similarity_threshold=0.5):
    triplets = []
    all_nodes = list(G.nodes())
    
    for anchor in G.nodes():
        positives = list(G.neighbors(anchor))
        if not positives:
            continue
        
        for _ in range(num_triplets_per_anchor):
            positive = random.choice(positives)
            # select max three positive elements
            potential_negatives = [n for n in all_nodes if n != anchor and n not in G.neighbors(anchor)]
            if not potential_negatives:
                continue
            
            # similarities = []
            # for neg in potential_negatives:
            #     sim = Edge(anchor, neg, st).get_table_similarity()
            #     if sim >= similarity_threshold:
            #         similarities.append((neg, sim))
            
            negative = random.choice(potential_negatives)
            # if not similarities:
            # else:
            #     similarities.sort(key=lambda x: x[1], reverse=True)
            #     hard_negatives = [neg for neg, sim in similarities[:top_k]]
            #     negative = random.choice(hard_negatives)
            triplet = InputExample(
                texts=[node_descriptions[anchor], node_descriptions[positive], node_descriptions[negative]]
            )
            triplets.append(triplet)
    
    return triplets

#triplet_examples = generate_triplets(G, node_descriptions)
