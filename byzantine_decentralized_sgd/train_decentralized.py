import argparse 
import numpy as np
import json
import torch
from torchvision import datasets, transforms

from _params import add_common_params, add_decentralized_params
from _train_utils import test, plot_learning_curve
from _node import Node
from _byzantine_node import ByzantineNode
from _data_utils import default_transform, MNISTSlice
from _logic import *
from _krum import krum, _distance
from _average import get_average_gradients, get_std_gradients
from _attack import setup_lp_norm_attack
from _trimmed_mean import trimmed_mean

        
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Decentralized Training')
    add_common_params(parser)
    add_decentralized_params(parser)
    args = parser.parse_args()
    
    use_multiprocess = not(args.no_multiprocess)
    
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
            node = ByzantineNode.create(mode=args.byzantine_mode)(            
                node_idx,
                node_trainset,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                momentum=args.momentum,
                log_interval=args.log_interval,
                mode=args.byzantine_mode)
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
    honest_nodes = [n for n in nodes if n.id not in byzantine_idx]

    print('Created {} Byzantine nodes: {}'.format(len(byzantine_idx), byzantine_idx))
    print('Done.')

    # decentralized training
    print('starting decentralized training...')
    consensus_w = honest_nodes[0].get_weights()
    align_all_nodes_to_consensus(nodes, consensus_w)

    learning_curve = []
    test_accuracy = []
    for i in range(args.epochs):
        print('epoch:: {} out of {}'.format(i + 1, args.epochs))

        while True:
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

            if (len(byzantine_participants_ids) < args.participants_size / 2) and (len(byzantine_committee_ids) < args.committee_size / 2):
                break

        participants = nodes[participant_ids]
        committee = nodes[committe_ids]
        
        print('training all nodes...')
        all_train_loss = run_all(participants, k=args.internal_epochs, multiprocess=use_multiprocess)
        avg_train_loss = np.mean([loss for id_, loss in all_train_loss if id_ not in byzantine_idx])

        # setting up the Lp-norm attack (if there are byzantines)
        if args.byzantine_mode == 'lp-norm':

            honest_participants = [n for n in participants if n.id not in byzantine_idx]
            mu = get_average_gradients(honest_participants)
            std = get_std_gradients(honest_participants)
            gamma = setup_lp_norm_attack(participants, byzantine_idx, mu, std, consensus_w, f=len(byzantine_participants_ids))
            print('Chosen Lp-norm attack gamma: {}'.format(gamma))

        if args.aggregator == 'union-consensus':

            print('collecting weights from participants...')
            w_array = collect_participants_weights(participants)

            print('collecting votes from committee...')
            votes = collect_committee_votes(committee, w_array, multiprocess=use_multiprocess)
            print("Votes:", dict([(k, participant_ids[v]) for k, v in votes.items()]))
            union_consensus, n_unique_recipients = reach_union_consensus(votes)
            union_consensus_ids = participant_ids[union_consensus]

            print('reached union consensous of size {}, with {} unique recipients'.format(
                len(union_consensus),
                n_unique_recipients))
            byzantine_consensus_ids = set(union_consensus_ids).intersection(byzantine_participants_ids)
            print('Consensus: {}, #Byzantine nodes inside: {} --> {}'.format(
                union_consensus_ids, len(byzantine_consensus_ids), byzantine_consensus_ids))

            consensus_w = get_average_union_consensus(w_array, union_consensus)
            align_all_nodes_to_consensus(nodes, consensus_w)

            learning_curve.append({
                'train_loss': avg_train_loss,
                'union_size': len(union_consensus),
                'n_unique_recipients': n_unique_recipients,
                'n_byzantine_participants': len(byzantine_participants_ids),
                'n_byzantine_committee': len(byzantine_committee_ids),
                'n_byzantine_consensus': len(byzantine_consensus_ids),
            })

        elif args.aggregator == 'krum':

            print('collecting gradients from participants and running krum...')
            krum_node_idx, krum_scores = krum(participants, f=len(byzantine_participants_ids))
            selected_node = participants[krum_node_idx]

            is_byzantine_selected = int(selected_node.id in byzantine_participants_ids)
            print('Selected node by krum: {}, is byzantine: {}'.format(selected_node.id, is_byzantine_selected))
            print('Krum selected score: {}'.format(krum_scores[krum_node_idx]))

            consensus_w = selected_node.get_weights()
            align_all_nodes_to_consensus(nodes, consensus_w)

            learning_curve.append({
                'train_loss': avg_train_loss,
                'selected_node': selected_node.id,
                'is_byzantine_selected': is_byzantine_selected,
            })

        elif args.aggregator == 'trimmed-mean':

            print('collecting gradients from participants and running trimmed mean...')
            trimmed_mean_grads = trimmed_mean(participants, f=len(byzantine_participants_ids))

            # simulate the step take by the trimmed mean gradient
            honest_participants = [n for n in participants if n.id not in byzantine_idx]
            proxy_node = honest_participants[0]
            proxy_node.set_weights(consensus_w)
            proxy_node.set_gradients(trimmed_mean_grads)
            proxy_node.take_step()

            consensus_w = proxy_node.get_weights()
            align_all_nodes_to_consensus(nodes, consensus_w)

            learning_curve.append({
                'train_loss': avg_train_loss
            })

        else:  # average

            print('collecting gradients from participants and running average...')
            average_grads = get_average_gradients(participants)

            # simulate the step take by the average gradient
            honest_participants = [n for n in participants if n.id not in byzantine_idx]
            proxy_node = honest_participants[0]
            proxy_node.set_weights(consensus_w)
            proxy_node.set_gradients(average_grads)
            proxy_node.take_step()

            consensus_w = proxy_node.get_weights()
            align_all_nodes_to_consensus(nodes, consensus_w)

            learning_curve.append({
                'train_loss': avg_train_loss
            })

        if args.byzantine_mode == 'lp-norm':
            learning_curve[-1]['gamma'] = gamma

        if i % 1 == 0:
            accuracy, popular_misses = test(
                args, participants[0]._model, participants[0]._device, test_loader)
            test_accuracy.append({'accuracy': accuracy, 'popular_misses': popular_misses})

    #plot_learning_curve(learning_curve, test_accuracy, n_participants=args.participants_size)
    
    with open('raw_learning_curve__{}.json'.format(args.aggregator), 'w') as f_raw:
        json.dump(
            {
                'setting': vars(args),
                'train': learning_curve,
                'evaluation': test_accuracy
            },
            f_raw
        )
        
if __name__ == '__main__':
    main()
