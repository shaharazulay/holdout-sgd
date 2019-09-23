import collections
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  
from matplotlib import pyplot as plt


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    popular_misses = collections.Counter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            miss_cond = (pred != target.view_as(pred))
            popular_misses.update(zip(
                [int(_) for _ in target.view_as(pred)[miss_cond].numpy()],
                [int(_) for _ in pred[miss_cond].numpy()]
            ))
                
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset), popular_misses.most_common(n=5)
        

def plot_learning_curve(learning_curve, test_acc, n_participants, output_dir='.'):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss [avg]', color=color)
    ax1.plot([c['train_loss'] for c in learning_curve], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:blue'
    ax2.set_ylabel('union consensus', color=color)
    ax2.plot([c['union_size'] for c in learning_curve], color=color)
    ax2.plot([c['n_unique_recipients'] for c in learning_curve], color='tab:green')
    ax2.set_ylim([0, n_participants])
    ax2.legend(['union size', '#recipients'])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title('Training Curve & Union Consensus')
    fig.savefig(output_dir + '/train_curve_and_union_consensus.jpg')
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss [avg]', color=color)
    ax1.plot([c['train_loss'] for c in learning_curve], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:blue'
    ax2.set_ylabel('byzantine', color=color)
    ax2.plot([c['n_byzantine_participants'] for c in learning_curve], color=color)
    ax2.plot([c['n_byzantine_committee'] for c in learning_curve], color='tab:green')
    ax2.plot([c['n_byzantine_consensus'] for c in learning_curve], color='tab:red')
    ax2.set_ylim([0, n_participants])
    ax2.legend(['# in participants', '# in committee', '# in union consensus'])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title('Training Curve & Byzantine Effect')
    fig.savefig(output_dir + '/train_curve_and_byzantine_effect.jpg')
    
    fig, ax1 = plt.subplots()
    ax1.plot([c['accuracy'] for c in test_acc])
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Acc [%]', color=color)
    plt.title('Test Accuray [%]')
    fig.savefig(output_dir + '/test_accuracy.jpg')