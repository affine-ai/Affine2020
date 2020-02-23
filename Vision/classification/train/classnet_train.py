from Affine.Vision.classification.src.darknet53 import darknet
from dataset_utils import load_imagenet_data as load_data, load_imagenet_val as load_val
from dataset_utils import data_prefetcher
from train_utils import parse_args, AverageMeter, ProgressMeter, Config, setup_and_launch

import os, time, datetime
import warnings

import math
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

HTIME = lambda t: time.strftime( "%H:%M:%S", time.gmtime( t ) )

def main_worker( gpu, args, config ):
    best_acc1 = 0

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    args.writer = None
    if args.gpu or gpu % args.gpus_per_node == 0:
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
                           lr=args.base_lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay )

    # Nvidia documentation states - 
    # "O2 exists mainly to support some internal use cases. Please prefer O1"
    # https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    model, optimizer = amp.initialize( model, optimizer, opt_level="O1" )

    if args.gpu is None:
        # By default, apex.parallel.DistributedDataParallel overlaps communication 
        # with computation in the backward pass.
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel( model )

    if args.resume:
        print( "Loading checkpoint {}".format( config.checkpoint_file ) )
        checkpoint = torch.load( config.checkpoint_file, map_location='cpu' )
        best_acc1 = checkpoint[ 'best_acc1' ]
        model.load_state_dict( checkpoint[ "model" ] )
        optimizer.load_state_dict( checkpoint[ "optimizer" ] )
        amp.load_state_dict( checkpoint[ "amp" ] )
        del checkpoint

    if args.evaluate:
        train_or_eval( False, gpu, val_loader, model, criterion, None, args, 0 )
        return


    start_epoch = args.start_epoch - 1
    end_epoch = start_epoch + args.epochs

    for epoch in range( start_epoch, end_epoch ):
        if args.gpu is None:
            train_loader.sampler.set_epoch( epoch )
        
        train_or_eval( True, gpu, train_loader, model, criterion, optimizer, args, epoch )

        if args.gpu or gpu % args.gpus_per_node == 0:
            acc1 = train_or_eval( False, gpu, val_loader, model, criterion, None, args, 0 )

            is_best = acc1 > best_acc1
            best_acc1 = max( acc1, best_acc1 )

            print( "Saving checkpoint")
            save_checkpoint( { "epoch"      : epoch + 1,
                               "model"      : model.state_dict(),
                               "optimizer"  : optimizer.state_dict(),
                               "amp"        : amp.state_dict(),
                               "best_acc1"  : best_acc1,
                            }, is_best, filename=config.checkpoint_write )
    if args.writer:
        args.writer.close()


def train_or_eval( train, gpu, loader, model, criterion, optimizer, args, epoch ):
    if train:
        model.train()
    else:
        model.eval()

    losses = AverageMeter( "Loss", ":.4e" )
    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    top5 = AverageMeter( "Accuracy5", ":6.2f" )
    prefix = "Epoch:[{}]".format( epoch + 1 ) if train else "Test: "
    progress = ProgressMeter( len( loader ), [ losses, top1, top5 ], prefix=prefix )

    prefetcher = data_prefetcher( loader )
    
    total_time = [ 0.0 ] * 5
    time0 = time_init = time.time()

    with torch.set_grad_enabled( mode=train ):
        for i, ( images, target ) in enumerate( prefetcher ):
            images = images.cuda( gpu, non_blocking=True )
            target = target.cuda( gpu, non_blocking=True )
            time_dataload = time.time()

            output = model( images )
            loss = criterion( output, target )
            time_forward = time.time()

            batch_size = images.size( 0 )
            acc1, acc5 = accuracy( output, target, topk=( 1, 5 ) )
            losses.update( loss.item(), batch_size )
            top1.update( acc1[ 0 ], batch_size )
            top5.update( acc5[ 0 ], batch_size )

            time_loss_update = time.time()
            time_backward = time.time()
            # All the code that needs to run only when training goes here
            if train:
                n_iter = epoch * len( loader ) + i
                lr = adjust_learning_rate( optimizer, n_iter, args, policy="triangle" )

                optimizer.zero_grad()
                
                with amp.scale_loss( loss, optimizer ) as scaled_loss:
                    scaled_loss.backward()
                
                optimizer.step()
                time_backward = time.time()

                if ( i % 100 == 0 ) and ( gpu % args.gpus_per_node == 0 ):
                    args.writer.add_scalar( "Loss/train/gpu{}".format( gpu ), loss.item(), n_iter )
                    args.writer.add_scalar( "Accuracy/train/gpu{}".format( gpu ), acc1, n_iter )
                    args.writer.add_scalar( "Loss/Accuracy", acc1, lr * 10000 )
            # End of training specific code
            time_misc = time.time()            

            time_markers = [ time_init, time_dataload, time_forward, time_loss_update, time_backward, time_misc ]
            time_delta = time_markers[ 1: ] - time_markers[ :-1 ]
            total_time += time_delta

            if  i % 200 == 0 and gpu % args.gpus_per_node == 0:
                progress.display( i )
                for x, y in ( ( "Time loading data:", HTIME( total_time[ 0 ] ) ),
                              ( "Time forward:", HTIME( total_time[ 1 ] ) ),
                              ( "Time loss update:", HTIME( total_time[ 2 ] ) ),
                              ( "Time backward:", HTIME( total_time[ 3 ] ) ),
                              ( "Time misc:", HTIME( total_time[ 4 ] ) )
                            ):
                    print( "{:<25s}{}".format( x, y ) )
                print()
            time_init = time.time()
            
    print( "Total epoch time: {}".format( HTIME( time.time() - time0 ) ) )
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

def adjust_learning_rate( optimizer, i, args, policy="triangle" ):
    """learning rate schedule
    """
    cycle = math.floor( 1 + i / ( 2 * args.stepsize ) )
    if policy is "triangle2":
        range = ( args.max_lr - args.base_lr ) / pow( 2, ( cycle - 1 ) )
    else:
        range = ( args.max_lr - args.base_lr )

    x = abs( i / args.stepsize - 2 * cycle + 1 )
    lr = args.base_lr + range * max( 0.0, ( 1.0 - x ) )

    for param_group in optimizer.param_groups:
        param_group[ 'lr' ] = lr
    return lr

def save_checkpoint( state, is_best=True, filename=None ):
    torch.save( state, filename )


if __name__ == "__main__":
    config = Config()
    config.train_path = "/home/vipul/Datasets/ImageNet/train"
    config.val_path = "/home/vipul/Datasets/ImageNet/val"
    config.checkpoint_path = "checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    setup_and_launch( worker_fn=main_worker, config=config )