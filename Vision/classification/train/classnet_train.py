#!/usr/bin/env python3

from Affine.Vision.classification.src.darknet53 import darknet
from dataset_utils import load_imagenet_data as load_data, load_imagenet_val as load_val
from dataset_utils import data_prefetcher
from train_utils import parse_args, AverageMeter, ProgressMeter, setup_and_launch, adjust_learning_rate

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


def main_worker( gpu, args, config, hyper ):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    best_acc1 = 0    
    args.writer = None
    distributed = not args.gpu

    if args.lr_policy not in ( "triangle", "triangle2" ):
        print( "Unsupported learning rate policy." )
        raise SystemExit

    if distributed:
        dist.init_process_group( backend=args.dist_backend, 
                                 init_method="tcp://10.0.1.164:12345", 
                                 world_size=args.world_size, rank=gpu )
        print( "Process: {}, rank: {}, world_size: {}".format( gpu, dist.get_rank(), dist.get_world_size() ) )

    # Set the default device, any tensors created by cuda by 'default' will use this device
    torch.cuda.set_device( gpu )

    train_loader = load_data( config.train_path, args, distributed )
    val_loader = load_val( config.val_path, args, distributed )
    assert train_loader.dataset.classes == val_loader.dataset.classes

    model = darknet()
    model.cuda( gpu )

    criterion = nn.CrossEntropyLoss().cuda( gpu )
    optimizer = optim.SGD( model.parameters(), 
                           lr=hyper.base_lr,
                           momentum=hyper.momentum,
                           weight_decay=hyper.weight_decay )

    # Nvidia documentation states - 
    # "O2 exists mainly to support some internal use cases. Please prefer O1"
    # https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    model, optimizer = amp.initialize( model, optimizer, opt_level="O1" )

    if distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication 
        # with computation in the backward pass.
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel( model )

    if args.resume:
        checkpoint = torch.load( config.checkpoint_file, map_location='cpu' )
        best_acc1 = checkpoint[ 'best_acc1' ]
        model.load_state_dict( checkpoint[ "model" ] )
        optimizer.load_state_dict( checkpoint[ "optimizer" ] )
        amp.load_state_dict( checkpoint[ "amp" ] )
        start_epoch = checkpoint[ "epoch" ]
        print( "Resuming from epoch {}, {}".format( start_epoch + 1, config.checkpoint_file ) )
        del checkpoint
    start_epoch = args.start_epoch - 1 if "start_epoch_user" in args.__dict__ else start_epoch

    if args.evaluate:
        train_or_eval( False, gpu, val_loader, model, criterion, None, args, hyper, 0 )
        return

    if not distributed or gpu % args.gpus_per_node == 0:
        args.writer = SummaryWriter( filename_suffix="{}".format( gpu ) )

    end_epoch = start_epoch + args.epochs
    for epoch in range( start_epoch, end_epoch ):
        if distributed:
            train_loader.sampler.set_epoch( epoch )
        train_or_eval( True, gpu, train_loader, model, criterion, optimizer, args, hyper, epoch )

        if gpu % args.gpus_per_node == 0 or not distributed:
            acc1 = train_or_eval( False, gpu, val_loader, model, criterion, None, args, hyper, 0 )

            is_best = acc1 > best_acc1
            best_acc1 = max( acc1, best_acc1 )

            print( "Saving model state" )
            save_checkpoint( { "epoch"      : epoch + 1,
                               "base_lr"    : hyper.base_lr,
                               "max_lr"     : hyper.max_lr,
                               "lr_policy"  : hyper.lr_policy,
                               "batch_size" : hyper.batch_size,
                               "model"      : model.state_dict(),
                               "optimizer"  : optimizer.state_dict(),
                               "amp"        : amp.state_dict(),
                               "best_acc1"  : best_acc1,
                             }, is_best, filename=config.checkpoint_write )
    if args.writer:
        args.writer.close()


def train_or_eval( train, gpu, loader, model, criterion, optimizer, args, hyper, epoch ):
    phase = "train" if train else "test"
    model.train() if train else model.eval()

    losses = AverageMeter( "Loss", ":.4e" )
    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    top5 = AverageMeter( "Accuracy5", ":6.2f" )
    prefix = "Epoch:[{}]".format( epoch + 1 ) if train else "Test: "
    progress = ProgressMeter( len( loader ), [ losses, top1, top5 ], prefix=prefix )


    # The timing infrastructure below is not very accurate relatively, due to asynchronous 
    # GPU operation. However, it is very lightweight and can be left working all the time 
    # for a relative estimation of how the training loop is performing.
    markers = ( "Time loading data:", "Time forward:", "Time backward:" )
    t = torch.zeros( [ len( markers ) + 1 ], dtype=float )
    total = torch.zeros( [ len( markers ) ], dtype=float )
    
    if args.prof:
        print( "Profiling started" )
        torch.cuda.cudart().cudaProfilerStart()

    t_init = t[0] = time.time()
    prefetcher = data_prefetcher( loader )
    with torch.set_grad_enabled( mode=train ):
        for i, ( images, target ) in enumerate( prefetcher ):
            t[1] = time.time()
            niter = epoch * len( loader ) + i

            if args.prof: torch.cuda.nvtx.range_push( "Prof start iteration {}".format( i ) )

            if args.prof: torch.cuda.nvtx.range_push( "forward" )
            output = model( images )
            if args.prof: torch.cuda.nvtx.range_pop()

            loss = criterion( output, target )

            t[2] = time.time()
            t[3] = time.time()
            
            if train:
                lr = adjust_learning_rate( optimizer, niter, hyper )

                optimizer.zero_grad()
                
                if args.prof: torch.cuda.nvtx.range_push( "backward" )
                with amp.scale_loss( loss, optimizer ) as scaled_loss:
                    scaled_loss.backward()
                if args.prof: torch.cuda.nvtx.range_pop()

                if args.prof: torch.cuda.nvtx.range_push( "optimizer step" )
                optimizer.step()
                if args.prof: torch.cuda.nvtx.range_pop()
                t[3] = time.time()

            total += t[ 1: ] - t[ :-1 ]

            publish_stats = i % 100 == 0 and gpu % args.gpus_per_node == 0
            if not train or publish_stats:
                acc1, acc5 = accuracy( output.detach(), target, topk=( 1, 5 ) )
                losses.update( loss.item(), images.size(0) )
                top1.update( acc1[0], images.size(0) )
                top5.update( acc5[0], images.size(0) )

            if publish_stats:
                progress.display( i )

            if  train and publish_stats:
                args.writer.add_scalar( "Loss/{}".format( phase ), loss.item(), niter )
                args.writer.add_scalar( "Accuracy/{}".format( phase ), acc1, niter )
                args.writer.add_scalar( "Loss/Accuracy", acc1, lr * 10000 )
                for x, y in zip( markers, total ):
                    print( "{:<25s}{}".format( x, HTIME( y ) ) )
                print()

            t[0] = time.time()
            if args.prof: torch.cuda.nvtx.range_pop()
            if args.prof and i == 20:
                print( "Profiling stopped" )
                torch.cuda.cudart().cudaProfilerStop()
                break

    print( "Total {} epoch time: {}".format( phase, HTIME( time.time() - t_init ) ) )
    return top1.avg

def accuracy( outputs, targets, topk=(1, ) ):
    with torch.no_grad():
        maxk = max( topk )
        batch_size = targets.size( 0 )
        _, topk_idx = outputs.topk( maxk, dim=1, largest=True, sorted=True )
        topk_idx = topk_idx.t()
        correct = topk_idx.eq( targets.expand_as( topk_idx ) )
        res = []
        for k in topk:
            correct_k = correct[ :k ].view( -1 ).float().sum( 0, keepdim=True )
            res.append( correct_k.mul_( 100.0 / batch_size ) )
    return res

def save_checkpoint( state, is_best=True, filename=None ):
    torch.save( state, filename )


if __name__ == "__main__":
    setup_and_launch( main_worker )