from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from _model import Net
from _params import add_common_params
from _train_utils import train, test
from _data_utils import default_transform


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Centralized Training')
    add_common_params(parser)
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=default_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
        
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=default_transform),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
