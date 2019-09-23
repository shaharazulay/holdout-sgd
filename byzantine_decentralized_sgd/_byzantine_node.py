import torch

from _node import Node
from _model import Net


class ByzantineNode(object):

    @staticmethod
    def create(mode='random'):
        if mode == 'random':
            return RandomByzantineNode
        return SwapByzantineNode
            
            
class RandomByzantineNode(Node):

    def __init__(
        self,
        node_id,
        train_set,
        batch_size=64,
        learning_rate=0.01,
        momentum=0.5,
        log_interval=10,
        mode=None):
        
        RandomByzantineNode._suffle_labels(train_set) # create byzantine effect
        
        super(RandomByzantineNode, self).__init__(
            node_id,
            train_set,
            batch_size,
            learning_rate,
            momentum,
            log_interval)
        
    def _train_one_epoch(self, verbose):
        self._randomize_weights()
        if verbose:
            print('Node {}:: Byzantine node. no training is performed'.format(self.id))
            
        return torch.Tensor([0.0]) # return zero loss
            
    def _randomize_weights(self):
        self._model = Net().to(self._device)
        
    @staticmethod
    def _suffle_labels(train_set):
        n = len(train_set.train_labels)
        train_set.train_labels = train_set.train_labels[torch.randperm(n)]
            

class SwapByzantineNode(Node):

    def __init__(
        self,
        node_id,
        train_set,
        batch_size=64,
        learning_rate=0.01,
        momentum=0.5,
        log_interval=10,
        mode='swap-0-1'):
        
        SwapByzantineNode._suffle_labels(train_set, mode) # create byzantine effect
        
        super(SwapByzantineNode, self).__init__(
            node_id,
            train_set,
            batch_size,
            learning_rate,
            momentum,
            log_interval)
        
    @staticmethod
    def _suffle_labels(train_set, mode):
        # mode = 'swap-{}-{}'
        label_a = int(mode.split('-')[1])
        label_b = int(mode.split('-')[2])
        train_set.train_labels[train_set.train_labels == label_a] = -1
        train_set.train_labels[train_set.train_labels == label_b] = label_a
        train_set.train_labels[train_set.train_labels == -1] = label_b