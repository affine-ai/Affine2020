from train_utils import parse_args, adjust_learning_rate
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18


dataset_size = 1281216

def plot_lr( optim, args ):
    lr_hist = []

    batch_size = args.batch_size
    n_per_epoch = int( dataset_size / batch_size )
    print( "number of iterations per epoch:{}".format( n_per_epoch ) )

    start_epoch = args.start_epoch - 1
    end_epoch = start_epoch + args.epochs

    for epoch in range( start_epoch, end_epoch ):
        for i in range( n_per_epoch ):
            niter = epoch * n_per_epoch + i
            lr = adjust_learning_rate( optim, niter, args, policy=args.lr_policy )
            lr_hist.append( lr )

    index = list( range( n_per_epoch * args.epochs ) )
    plt.plot( index, lr_hist )
    plt.show()

# create a dummy optimizer
args = parse_args()
optim = torch.optim.SGD( resnet18().parameters(), lr=args.base_lr )
plot_lr( optim, args )