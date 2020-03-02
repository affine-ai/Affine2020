from Affine.Vision.classification.src.darknet53 import darknet
from dataset_utils import load_imagenet_data as load_data, load_imagenet_val as load_val
from dataset_utils import data_prefetcher
from train_utils import parse_args, AverageMeter, ProgressMeter, Config, load_checkpoint

import os, time, datetime
import warnings
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision


HTIME = lambda t: time.strftime( "%H:%M:%S", time.gmtime( t ) )

score = torch.zeros( [ 1000 ], dtype=int )
present = torch.zeros( [ 1000 ], dtype=int )

def validate( loader, model, args ):
    global score, present

    if args.gpu is not None:
        print( "** Running on GPU{}".format( args.gpu ) )
        torch.cuda.set_device( args.gpu )
        model.cuda( args.gpu )
        score = score.cuda( args.gpu )
        present = present.cuda( args.gpu )
    else:
        model.eval().cpu()

    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    prefix = "Epoch:[{}]".format( 1 )
    progress = ProgressMeter( len( loader ), [ top1 ], prefix=prefix )

    with torch.set_grad_enabled( mode=False ):
        for i, ( images, targets ) in enumerate( loader ):
            if args.gpu is not None:
                images = images.cuda( args.gpu, non_blocking=True )
                targets = targets.cuda( args.gpu, non_blocking=True )

            t0 = time.time()
            output = model( images )
            t1 = time.time()
            accuracy_with_score( output, targets )
            t2 = time.time()

            if  i % 50 == 0:
                progress.display( i )
                print( "{:<20s}\t{}".format( "forward", HTIME( t1 - t0 ) ) )
                print( "{:<20s}\t{}".format( "accuracy", HTIME( t2 - t1 ) ) )

    non_zero = present.nonzero().squeeze( 1 )

    present = present[ non_zero ]
    score = score[ non_zero ]
    score = score.float().div( present )
    score = score.mul_( 100 )

    
    print( "\nBest scores" )
    best, ids = score.topk( 10, dim=0, largest=True, sorted=True )
    for i, p in zip( ids, best ):
        print( "{:<10d}\t{:4.1f}".format( i.data, p.data ) )

    print( "\nWorst scores" )
    last, ids = score.topk( 10, dim=0, largest=False, sorted=True )
    for i, p in zip( ids, last ):
        print( "{:<10d}\t{:4.1f}".format( i.data, p.data ) )


def accuracy_with_score( outputs, targets ):
    global present, score
    with torch.no_grad():
        i, c = torch.unique( targets, return_counts=True )
        present[ i ] += c

        _, idx = outputs.topk( 1, dim=1, largest=True, sorted=True )
        idx = idx.t()
        correct = idx.eq( targets.expand_as( idx ) )

        positives = torch.masked_select( idx, correct )
        i, c = torch.unique( positives, return_counts=True )
        score[ i ] += c


def main():
    args = parse_args()
    loader = load_val( config.val_path, args, distributed=False )

    model = darknet().eval()
    #model = torchvision.models.resnet101( pretrained=True ).eval()

    checkpoint_path = os.path.join( config.checkpoint_path, config.checkpoint_name )
    load_checkpoint( model, checkpoint_path )
    validate( loader, model, args )


if __name__ == "__main__":
    config = Config()
    config.val_path = "/home/vipul/Datasets/ImageNet/val"
    config.checkpoint_path = "checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    main()