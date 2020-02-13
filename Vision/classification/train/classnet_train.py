from Affine.Vision.classification.src.darknet53 import darknet
from dataset_utils import load_imagenet_data as load_data, load_imagenet_val as load_val
from train_utils import parse_args, AverageMeter, ProgressMeter, Config, setup_and_launch

import os, time, datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.distributed as dist
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

try:
    import apex
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


def main_worker( gpu, args, config ):
    best_acc1 = 0
    args.writer = SummaryWriter( filename_suffix="{}".format( gpu ) )

    if args.gpu is None:
        dist.init_process_group( backend=args.dist_backend, 
                                 init_method="tcp://10.0.1.164:12345", 
                                 world_size=args.world_size, rank=gpu )
        print( "Process: {}, rank: {}, world_size: {}".format( gpu, dist.get_rank(), dist.get_world_size() ) )

    # Set the default device, any tensors created by cuda by 'default' will use this device
    torch.cuda.set_device( gpu )

    train_loader = load_data( config, args, gpu )
    val_loader = load_val( config, args )
    assert train_loader.dataset.classes == val_loader.dataset.classes

    model = darknet()
    model.cuda( gpu )

    criterion = nn.CrossEntropyLoss().cuda( gpu )
    optimizer = optim.SGD( model.parameters(), 
                           lr=args.learning_rate, 
                           momentum=args.momentum, 
                           weight_decay=args.weight_decay )

    # Nvidia documentation states - 
    # "O2 exists mainly to support some internal use cases. Please prefer O1"
    # https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    model, optimizer = amp.initialize( model, optimizer, 
                                       opt_level="O1" )

    if args.gpu is None:
        model = apex.parallel.DistributedDataParallel( model )

    if args.resume:
        print( "Loading checkpoint {}".format( config.checkpoint_file ) )
        checkpoint = torch.load( config.checkpoint_file, map_location='cpu' )
        best_acc1 = checkpoint[ 'best_acc1' ]
        model.load_state_dict( checkpoint[ "model" ] )
        optimizer.load_state_dict( checkpoint[ "optimizer" ] )
        amp.load_state_dict( checkpoint[ "amp" ] )

    if args.evaluate:
        validate( val_loader, model, criterion, gpu, args )
        return

    ################################
    # Main training loop starts here
    ################################
    time0 = time.time()
    for epoch in range( args.start_epoch, args.epochs ):
        if args.gpu is None:
            train_loader.sampler.set_epoch( epoch )
        
        train( train_loader, model, criterion, optimizer, epoch, gpu, args )

        print( "Training time: {}".format( datetime.timedelta( seconds=time.time() - time0 ) ) )

        acc1 = validate( val_loader, model, criterion, gpu, args )
        is_best = acc1 > best_acc1
        best_acc1 = max( acc1, best_acc1 )

        if args.gpu or ( gpu % args.world_size == 0 ):
            print( "Saving checkpoint")
            save_checkpoint( { "epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(),
                               "amp": amp.state_dict(),
                               "best_acc1": best_acc1,
                             }, is_best, filename=config.checkpoint_write )

        time0 = time.time()
    
    args.writer.close()

def train( loader, model, criterion, optimizer, epoch, gpu, args ):
    model.train()
    train_or_eval( True, gpu, loader, model, criterion, optimizer, args, epoch )

def validate( loader, model, criterion, gpu, args ):
    model.eval()
    # All GPUs get the same copy of the model. Therefore, 
    # running validation on one of them should be sufficient.
    if gpu % args.gpus_per_node == 0:
        return train_or_eval( False, gpu, loader, model, criterion, None, args, 0 )
    return 0

def train_or_eval( train, gpu, loader, model, criterion, optimizer, args, epoch ):
    losses = AverageMeter( "Loss", ":.4e" )
    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    top5 = AverageMeter( "Accuracy5", ":6.2f" )
    n_inputs = len( loader )

    prefix = "Epoch:[{}]".format( epoch + 1 ) if train else "Test: "
    progress = ProgressMeter( n_inputs, [ losses, top1, top5 ], prefix=prefix )

    with torch.set_grad_enabled( mode=train ):
        for i, ( images, target ) in enumerate( loader ):
            images = images.cuda( gpu, non_blocking=True )
            target = target.cuda( gpu, non_blocking=True )
            output = model( images )
            loss = criterion( output, target )

            batch_size = images.size( 0 )
            acc1, acc5 = accuracy( output, target, topk=( 1, 5 ) )
            losses.update( loss.item(), batch_size )
            top1.update( acc1[ 0 ], batch_size )
            top5.update( acc5[ 0 ], batch_size )

            if i % 100 == 0:
                progress.display( i )
            
            # All the code that needs to run only when training goes here
            if train:
                optimizer.zero_grad()
                with amp.scale_loss( loss, optimizer ) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    n_iter = n_inputs * epoch + i
                    args.writer.add_scalar( "Loss/train/gpu{}".format( gpu ), loss.item(), n_iter )
                    args.writer.add_scalar( "Accuracy/train/gpu{}".format( gpu ), acc1, n_iter )
            # End of training specific code
    return top1.avg

def accuracy( outputs, targets, topk=(1, ) ):
    with torch.no_grad():
        maxk = max( topk )
        batch_size = targets.size( 0 )
        _, topk_classes = outputs.topk( maxk, dim=1, largest=True, sorted=True )
        topk_classes = topk_classes.t()
        correct = topk_classes.eq( targets.expand_as( topk_classes ) )

        res = []
        for k in topk:
            correct_k = correct[ :k ].view( -1 ).float().sum( 0, keepdim=True )
            res.append( correct_k.mul_( 100.0 / batch_size ) )
        return res

def adjust_learning_rate( optimizer, epoch, args ):
    """learning rate drops by 1/10th every 30 epochs
    """
    lr = args.learning_rate * ( 0.1 ** ( epoch // 30 ) )
    for param_group in optimizer.param_groups:
        param_group[ 'lr' ] = lr

def save_checkpoint( state, is_best=True, filename=None ):
    torch.save( state, filename )


if __name__ == "__main__":
    config = Config()
    config.train_path = "/home/vipul/Datasets/ImageNet/train"
    config.val_path = "/home/vipul/Datasets/ImageNet/val"
    config.checkpoint_path = "checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    setup_and_launch( worker_fn=main_worker, config=config )