import argparse 
import numpy as np
import torch
from torchvision import datasets, transforms

from _params import add_common_params, add_decentralized_params
from _train_utils import test, plot_learning_curve
from _node import Node
from _byzantine_node import ByzantineNode
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

    n_byzantine = int(args.nodes * args.byzantine)
    
    nodes = []
    byzantine_idx = []
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

        if node_idx < n_byzantine:  # node was chosen as byzantine node
            byzantine_idx.append(node_idx)
            node = ByzantineNode(
                node_idx,
                node_trainset,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                momentum=args.momentum,
                log_interval=args.log_interval)
        else:
            node = Node(
                node_idx,
                node_trainset,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                momentum=args.momentum,
                log_interval=args.log_interval)
        
        nodes.append(node)
    
    nodes = np.array(nodes)
    print('Created {} Byzantine nodes: {}'.format(len(byzantine_idx), byzantine_idx))
    print('Done.')

    # decentralized training
    print('starting decentralized training...')
    
    learning_curve = []
    test_accuracy = []
    for i in range(args.epochs):
        print('epoch:: {} out of {}'.format(i + 1, args.epochs))
        
        participant_ids = select_participants(
            n_nodes=args.nodes,
            n_participants=args.participants_size)

        committe_ids = select_committee(
            n_nodes=args.nodes,
            n_committee=args.committee_size,
            exclude=participant_ids)
            
        byzantine_participants_ids = set(participant_ids).intersection(set(byzantine_idx))
        print('{} byzantine participants selected...'.format(len(byzantine_participants_ids)))
        
        byzantine_committee_ids = set(committe_ids).intersection(set(byzantine_idx))
        print('{} byzantine committe selected...'.format(len(byzantine_committee_ids)))
        
        participants = nodes[participant_ids]
        committee = nodes[committe_ids]
        
        avg_train_loss = run_all(nodes, k=args.internal_epochs)
        
        print('collecting weights from participants...')
        w_array = collect_participants_weights(participants)

        print('collecting votes from committee...')
        votes = collect_committee_votes(committee, w_array)
        print("Votes:", votes)
        
        union_consensus, n_unique_recipients = reach_union_consensus(votes)
        union_consensus_ids = participant_ids[union_consensus]
        
        print('reached union consensous of size {}, with {} unique recipients'.format(
            len(union_consensus),
            n_unique_recipients))
        byzantine_consensus_ids = set(union_consensus_ids).intersection(byzantine_participants_ids)
        print('Consensus: {}, #Byzantine nodes inside: {} --> {}'.format(
            union_consensus_ids, len(byzantine_consensus_ids), byzantine_consensus_ids))
        
        # DEBUG ###
        if len(byzantine_consensus_ids) > 0:  # <<<
            print(
                "DEBUG: ",
                 nodes[-1]._calc_loss(),
                 nodes[-1]._calc_loss(nodes[list(byzantine_consensus_ids)[0]].get_weights()),
                 nodes[-1]._calc_loss(nodes[list(set(byzantine_idx) - byzantine_consensus_ids)[0]].get_weights())
            )  # <<<
        
        consensus_w = get_average_union_consensus(w_array, union_consensus)
        
        # DEBUG ###
        print("Losses before consensus ",[n._calc_loss() for n in nodes[-3:]])
        print("Losses after consensus ",[n._calc_loss(consensus_w) for n in nodes[-3:]])
        
        align_all_nodes_to_consensus(nodes, consensus_w)
        
        if i % 2 == 0:
            accuracy = test(args, participants[0]._model, participants[0]._device, test_loader)
            test_accuracy.append(accuracy)

        learning_curve.append({
            'train_loss': avg_train_loss,
            'union_size': len(union_consensus),
            'n_unique_recipients': n_unique_recipients,
            'n_byzantine_participants': len(byzantine_participants_ids),
            'n_byzantine_committee': len(byzantine_committee_ids),
            'n_byzantine_consensus': len(byzantine_consensus_ids)
        })
    
    print(test_accuracy)
    print(learning_curve)
    plot_learning_curve(learning_curve, test_accuracy, n_participants=args.participants_size)
        
if __name__ == '__main__':
    main()
