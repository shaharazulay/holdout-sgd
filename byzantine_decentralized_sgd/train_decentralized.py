import argparse 
import numpy as np
import torch
from torchvision import datasets, transforms

from _params import add_common_params, add_decentralized_params
from _train_utils import test
from _node import Node
from _data_utils import default_transform, MNISTSlice
from _logic import *

        
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Decentralized Training')
    add_common_params(parser)
    add_decentralized_params(parser)
    args = parser.parse_args()
    
    trainset_full = datasets.MNIST('../data', train=True, download=True, transform=default_transform)
    
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=default_transform),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    train_labels = trainset_full.train_labels.numpy()
    train_data = trainset_full.train_data.numpy()
    train_label_indices = {}
    
    # distribute data accross nodes
    print('setting up the simulation:: creating {} distributed nodes...'.format(args.nodes))
    
    for digit in range(10):
        train_label_indices[digit] = np.where(train_labels == digit)[0]

    nodes = []
    for node_idx in range(args.nodes):
        node_indices = []
        for digit in range(10):
            node_indices.extend(np.random.choice(
                train_label_indices[digit], 
                size=int(args.sample_size / 10))) # sample randomly from each label
                
        node_data = torch.from_numpy(train_data[node_indices])
        node_labels = torch.from_numpy(train_labels[node_indices])

        node_trainset = MNISTSlice(
            root='../data', 
            data=node_data, 
            labels=node_labels, 
            train=True, 
            transform=default_transform, 
            download=True)

        node = Node(
            node_idx,
            node_trainset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            momentum=args.momentum,
            log_interval=args.log_interval)
        
        nodes.append(node)
    
    nodes = np.array(nodes)
    print('Done.')

    # decentralized training
    print('starting decentralized training...')
    
    for i in range(args.epochs):
        print('epoch:: {} out of {}'.format(i + 1, args.epochs))
        
        participant_ids = select_participants(
            n_nodes=args.nodes,
            n_participants=args.participants_size)

        committe_ids = select_committee(
            n_nodes=args.nodes,
            n_committee=args.committee_size,
            exclude=participant_ids)

        participant_ids = committe_ids = range(6) ## <<<<<
        participants = nodes[participant_ids]
        committee = nodes[committe_ids]
        
        run_all(nodes, k=args.internal_epochs)
        
        print('collecting weights from participants...')
        w_array = collect_participants_weights(participants)

        print('collecting votes from committee...')
        print("W", [w['conv1.weight'][0, 0, 0, :2] for w in w_array]) # >>>
        votes = collect_committee_votes(committee, w_array)
        print(participant_ids, committe_ids)
        print(votes)
        
        union_consensus = reach_union_consensus(votes)
        print('reached union consensous of size {}'.format(len(union_consensus)))
        print(union_consensus)
        consensus_w = get_average_union_consensus(w_array, union_consensus)
        
        print("A ",[n._calc_loss(w_array[0]) for n in nodes[:5]])
        print("B ",[n._calc_loss(consensus_w) for n in nodes[:5]])

        align_all_nodes_to_consensus(nodes, consensus_w)
        
        if i % 10 == 3:
            test(args, participants[0]._model, participants[0]._device, test_loader)

        
if __name__ == '__main__':
    main()
