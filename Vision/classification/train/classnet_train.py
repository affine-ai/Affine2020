from Affine.Common.utils.src.train_utils import parse_args, AverageMeter, ProgressMeter, Config, setup_and_launch
from Affine.Vision.classification.src.darknet53 import Darknet53

import time
import os
import warnings
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.distributed as dist
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

def main_worker( gpu, args, config ):
    best_acc1 = 0
    args.writer = SummaryWriter( filename_suffix="{}".format( gpu ) )

    if args.gpu is None:
        dist.init_process_group( backend=args.dist_backend, 
                                 init_method="tcp://10.0.1.164:12345", 
                                 world_size=args.world_size, rank=gpu )
        print( "Process: {}, rank: {}, world_size: {}".format( gpu, dist.get_rank(), dist.get_world_size() ) )

    # Set the default GPU, any tensors created on the 'default' GPU will use this device
    torch.cuda.set_device( gpu )

    # Training set preprocessing and loader
    normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                      [ 0.229, 0.224, 0.225 ] )
    
    transform = transforms.Compose( [ transforms.RandomResizedCrop( 224 ), \
                                      transforms.RandomHorizontalFlip(), \
                                      transforms.ToTensor(),
                                      normalize ] )

    dataset = datasets.ImageFolder( config.train_path, transform=transform )

    if args.gpu is None:
        train_sampler = torch.utils.data.distributed.DistributedSampler( dataset, rank=gpu )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader( dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=( train_sampler is None ),
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                sampler=train_sampler )

    # Validation set preprocessing and loader
    val_transform = transforms.Compose( [ transforms.Resize( 224 ),
                                          transforms.CenterCrop( 224 ),
                                          transforms.ToTensor(),
                                          normalize ] )
    
    valset = datasets.ImageFolder( config.val_path, transform=val_transform )
    val_loader = torch.utils.data.DataLoader( valset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False, 
                                              num_workers=args.workers, 
                                              pin_memory=True )


    #model = torchvision.models.resnet50( pretrained=args.pretrained )
    model = Darknet53()
    criterion = nn.CrossEntropyLoss().cuda( gpu )
    optimizer = optim.SGD( model.parameters(), 
                           lr=args.learning_rate, 
                           momentum=args.momentum, 
                           weight_decay=args.weight_decay )

    model.cuda( gpu )
    if args.gpu is None:
        model = torch.nn.parallel.DistributedDataParallel( model, device_ids=[ gpu ], output_device=gpu )

    if args.resume:
        print( "Loading checkpoint {}".format( config.checkpoint_file ) )
        checkpoint = torch.load( config.checkpoint_file, map_location='cpu' )
        best_acc1 = checkpoint[ 'best_acc1' ]
        model.load_state_dict( checkpoint[ "model_state_dict" ] )
        optimizer.load_state_dict( checkpoint[ "optimizer_state_dict" ] )

    if args.evaluate:
        validate( val_loader, model, criterion, gpu, args )
        return

    # Main training loop starts here
    time0 = time.time()
    for epoch in range( args.start_epoch, args.epochs ):
        if args.gpu is None:
            train_sampler.set_epoch( epoch )
        
        train( train_loader, model, criterion, optimizer, epoch, gpu, args )

        print( "Training time: {}".format( datetime.timedelta( seconds=time.time() - time0 ) ) )

        acc1 = validate( val_loader, model, criterion, gpu, args )
        is_best = acc1 > best_acc1
        best_acc1 = max( acc1, best_acc1 )

        if args.gpu or ( gpu % args.world_size == 0 ):
            print( "Saving checkpoint")
            save_checkpoint( {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer_state_dict": optimizer.state_dict(),
            }, is_best, filename=config.checkpoint_write )

        time0 = time.time()
    
    args.writer.close()

def train( loader, model, criterion, optimizer, epoch, gpu, args ):
    model.train()
    train_or_eval( True, gpu, loader, model, criterion, optimizer, args, epoch )

def validate( loader, model, criterion, gpu, args ):
    model.eval()
    # All GPUs get the same copy of the model. There is no point in running
    # validation on multiple GPUs
    if gpu % args.gpus_per_node == 0:
        return train_or_eval( False, gpu, loader, model, criterion, None, args, 0 )
    return 0

def train_or_eval( train, gpu, loader, model, criterion, optimizer, args, epoch ):
    losses = AverageMeter( "Loss", ":.4e" )
    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    top5 = AverageMeter( "Accuracy5", ":6.2f" )
    n_inputs = len( loader )

    prefix = "Epoch:[{}]".format( epoch + 1 ) if train else "Test: "
    progress = ProgressMeter( n_inputs, \
                              [ losses, top1, top5 ], \
                              prefix=prefix )

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

            if i % 50 == 0:
                progress.display( i )
            
            # All the code that needs to run only when training goes here
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
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