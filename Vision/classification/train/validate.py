from Affine.Vision.classification.src.darknet53 import darknet
from dataset_utils import load_imagenet_data as load_data, load_imagenet_val as load_val
from dataset_utils import data_prefetcher
from train_utils import parse_args, AverageMeter, ProgressMeter, Config

import os, time, datetime
import warnings
import torch
import torch.nn as nn

def validate( loader, model, args ):
    model.eval()

    top1 = AverageMeter( "Accuracy1", ":6.2f" )
    prefix = "Epoch:[{}]".format( 1 )
    progress = ProgressMeter( len( loader ), [ top1 ], prefix=prefix )

    with torch.set_grad_enabled( mode=False ):
        for i, ( images, target ) in enumerate( loader ):

            output = model( images )
            acc1 = accuracy_with_score( output.detach(), target )
            top1.update( acc1[0], images.size(0) )

            if  i % 50 == 0:
                progress.display( i )


def accuracy_with_score( outputs, targets ):
    with torch.no_grad():
        b = targets.size( 0 )
        _, idx = outputs.topk( 1, dim=1, largest=True, sorted=True )
        idx = idx.t()
        correct = idx.eq( targets.expand_as( idx ) )
        
        correct_1 = correct[ :1 ].view( -1 ).float().sum( 0, keepdim=True )
        return correct_1.mul_( 100.0 / b )

def main():
    args = parse_args()
    checkpoint = os.path.join( config.checpoint_path, config.checkpoint_name )
    loader = load_val( config.val_path, args, distributed=False )

    model = darknet()

    print( "Loading checkpoint {}".format( checkpoint ) )
    checkpoint = torch.load( checkpoint, map_location='cpu' )
    model.load_state_dict( checkpoint[ "model" ] )
    del checkpoint


if __name__ == "__main__":
    config = Config()
    config.val_path = "/home/vipul/Datasets/ImageNet/val"
    config.checkpoint_path = "checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    main()