import torch

from _node import Node
from _model import Net


class ByzantineNode(Node):

    def __init__(
        self,
        node_id,
        train_set,
        batch_size=64,
        learning_rate=0.01,
        momentum=0.5,
        log_interval=10):
        
        ByzantineNode._suffle_labels(train_set) # create byzantine effect
        
        super(ByzantineNode, self).__init__(
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