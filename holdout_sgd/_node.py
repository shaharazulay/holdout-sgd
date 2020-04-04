import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from _model import Net


class Node(object):

    def __init__(
            self,
            node_id,
            train_set,
            batch_size=64,
            learning_rate=0.01,
            momentum=0,
            log_interval=10):

        self.id = node_id
        self._log_interval = log_interval

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self._train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)

        self._model = Net().to(self._device)
        self._optimizer = optim.SGD(self._model.parameters(), lr=learning_rate, momentum=momentum)

    def train_one_epoch(self):
        self._model.train()

        data, target = next(iter(self._train_loader))  # train for single batch
        data, target = data.reshape(-1, 28*28).to(self._device), target.to(self._device)

        self._optimizer.zero_grad()
        output = self._model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        self._optimizer.step()

        grads = []
        for param in self._model.parameters():
            grads.append(param.grad.view(-1))
        self._grads = torch.cat(copy.deepcopy(grads))

        return loss  # most updated loss (over a single batch)

    def _calc_loss(self, w=None):
        self._model.eval()

        w_before = self.get_weights()
        if w is not None:
            self.set_weights(w)  # change weights to suggested weights w

        total_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self._train_loader):
                data, target = data.reshape(-1, 28*28).to(self._device), target.to(self._device)
                output = self._model(data)
                total_loss += torch.nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss

        total_loss /= len(self._train_loader.dataset)

        self.set_weights(w_before)  # return to original weights
        return total_loss

    def vote(self, w_array, f):
        loss_array = []
        for w in w_array:
            loss_w = self._calc_loss(w)
            loss_array.append(loss_w)

        sorted_w_indices = np.argsort(loss_array)
        num_items_to_vote = len(w_array) - f
        return sorted_w_indices[:num_items_to_vote]

    def set_weights(self, w):
        self._model.load_state_dict(copy.deepcopy(w))

    def get_weights(self):
        return copy.deepcopy(self._model.state_dict())

    def get_gradients(self):
        return self._grads

    def set_gradients(self, grads):
        idx = 0
        grads_ = grads.clone()
        for param in self._model.parameters():
            size_layer = len(param.grad.view(-1))
            grads_layer = torch.Tensor(grads_[idx: idx + size_layer]).reshape_as(param.grad).detach()
            param.grad = grads_layer
            idx += size_layer

        self._grads = grads_

    def take_step(self):
        self._model.train()
        self._optimizer.step()
        self._optimizer.zero_grad()
