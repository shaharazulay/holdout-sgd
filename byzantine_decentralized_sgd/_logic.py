import numpy as np
from collections import Counter
from functools import reduce
import progressbar

from _parallel import run_in_parallel, async_run_in_parallel


def select_participants(n_nodes ,n_participants):
    return np.random.choice(n_nodes, size=n_participants, replace=False)
    
def select_committee(n_nodes, n_committee, exclude=[]):
    node_pool = list(set(range(n_nodes)) - set(exclude))
    return np.random.choice(node_pool, size=n_committee, replace=False)


def _run_one(node, k, verbose):
    loss = node.train_k_epochs(k=k, verbose=verbose)
    return loss.item()

def run_all(nodes, k=1):
    args_list = [(n, k, True) for n in nodes]
    train_loss = [_run_one(args) for args in args_list]  ##
    #train_loss = async_run_in_parallel(_run_one, args_list)
    return np.mean(train_loss)
    
    
def collect_participants_weights(participants):
    return np.array([p.get_weights() for p in participants])


def _collect_one(c, w_array):
    return c.id, c.vote(w_array)
    
def collect_committee_votes(committee, w_array):
    args_list = [(c, w_array) for c in committee]
    votes_list = async_run_in_parallel(_collect_one, args_list)
    return dict(votes_list)
    
    
def reach_union_consensus(votes, portion=2/3):
    vote_values = list(votes.values())
    n_votes_per_memeber = len(vote_values[0])
    
    flattened_votes = np.concatenate(vote_values).ravel()
    
    n_unique_recipients = len(np.unique(flattened_votes))
    vote_counts = Counter(flattened_votes)
    
    consensus_threshold = int(n_votes_per_memeber * portion)
    union_consensus = [
        vote for vote, count in vote_counts.items()
        if count > consensus_threshold]
    return union_consensus, n_unique_recipients
    
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