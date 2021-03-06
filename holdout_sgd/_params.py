def add_common_params(parser):
    parser.add_argument(
        '--batch-size',
        type=int, 
        default=64, 
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size', 
        type=int, 
        default=1000, 
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20, 
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.01, 
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0,
        help='SGD momentum (default: 0)')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', 
        type=int, 
        default=10, 
        help='how many batches to wait before logging training status')    
    parser.add_argument(
        '--save-model', 
        action='store_true', 
        default=False,
        help='For Saving the current Model')
                        
def add_decentralized_params(parser):
    parser.add_argument(
        '--nodes',
        type=int, 
        default=100, 
        help='number of nodes in decentralized setting (default: 1000)')
    parser.add_argument(
        '--sample-size',
        type=int, 
        default=1000, 
        help='number of samples per node (default: 100)')
    parser.add_argument(
        '--committee-size',
        type=int, 
        default=30, 
        help='number of nodes in committee (default: 100)')
    parser.add_argument(
        '--participants-size',
        type=int, 
        default=30, 
        help='number of nodes selected as participants (default: 100)')        
    parser.add_argument(
        '--byzantine',
        type=float, 
        default=0.0, 
        help='% of actual byzantine nodes inside the node pool (default: 0, range: 0-1')
    parser.add_argument(
        '--expected-byzantine',
        type=float,
        default=0.0,
        help='% of expected byzantine nodes inside the node pool (default: 0, range: 0-1')
    parser.add_argument(
        '--byzantine-mode',
        type=str, 
        default='random', 
        help='type of byzantine effect to be used (random, label swap, np-norm) (default: random, in case of swap enter the format swap-{}-{}.')
    parser.add_argument(
        '--no-multiprocess',
        action='store_true',
        default=False,
        help='turn off multiprocess mode (default: False)')
    parser.add_argument(
        '--aggregator',
        type=str,
        default='union-consensus',
        choices=['union-consensus', 'krum', 'average', 'trimmed-mean'],
        help='Choice of aggregator (default: union-consensus)')
