import numpy as np
from collections import Counter
from functools import reduce


def select_participants(n_nodes ,n_participants):
    return np.random.choice(n_nodes, size=n_participants, replace=False)
    
def select_committee(n_nodes, n_committee, exclude=[]):
    node_pool = list(set(range(n_nodes)) - set(exclude))
    return np.random.choice(node_pool, size=n_committee, replace=False)
    
def run_all(nodes, k=1):
    [n.train_k_epochs(k=k) for n in nodes]
    
def collect_participants_weights(participants):
    return np.array([p.get_weights() for p in participants])
    
def collect_committee_votes(committee, w_array):
    votes = {}
    for c in committee:
        votes[c.id] = c.vote(w_array)
    return votes
    
def reach_union_consensus(votes, portion=2/3):
    vote_values = list(votes.values())
    n_votes_per_memeber = len(vote_values[0])
    
    flattened_votes = np.concatenate(vote_values).ravel()
    vote_counts = Counter(flattened_votes)
    
    consensus_threshold = int(n_votes_per_memeber * portion)
    union_consensus = [
        vote for vote, count in vote_counts.items()
        if count > consensus_threshold]
    return union_consensus
    
def get_average_union_consensus(w_array, union_consensus):
    
    n_elements = len(union_consensus)
    
    def _reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value / n_elements
        return accumulator
        
    w_collection = w_array[union_consensus]
    consensus_w = reduce(_reducer, w_collection, {})
    return consensus_w


def align_all_nodes_to_consensus(nodes, consensus_w):
    [n.set_weights(consensus_w) for n in nodes]
    

def _reducer(accumulator, element):
    for key, value in element.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator