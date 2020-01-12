import torch

from _node import Node
from _model import Net
from _krum import krum


class ByzantineNode(object):

    @staticmethod
    def create(mode='random'):
        if mode == 'random':
            return RandomByzantineNode
        elif mode == 'lp-norm':
            return LpNormByzantineNode
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
        # mode = 'swap-{}-{}' or 'swap-all'
        def _swap_labels(train_set, label_a, label_b):
            train_set.train_labels[train_set.train_labels == label_a] = -1
            train_set.train_labels[train_set.train_labels == label_b] = label_a
            train_set.train_labels[train_set.train_labels == -1] = label_b

        if mode == 'swap-all':
            _swap_labels(train_set, 0, 9)
            _swap_labels(train_set, 1, 8)
            _swap_labels(train_set, 2, 7)
            _swap_labels(train_set, 3, 6)
            _swap_labels(train_set, 4, 5)
        else:
            label_a = int(mode.split('-')[1])
            label_b = int(mode.split('-')[2])
            _swap_labels(train_set, label_a, label_b)


class LpNormByzantineNode(Node):

    def __init__(
            self,
            node_id,
            train_set,
            batch_size=64,
            learning_rate=0.01,
            momentum=0.5,
            log_interval=10,
            mode=None):

        super(LpNormByzantineNode, self).__init__(
            node_id,
            train_set,
            batch_size,
            learning_rate,
            momentum,
            log_interval)

    def setup_attack(self, mu, std, gamma):
        B = mu.clone()
        B += gamma * std.clone()
        self.set_gradients(B)





