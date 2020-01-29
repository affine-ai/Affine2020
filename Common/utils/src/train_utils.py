import os
import argparse
import random
import scipy.io
import shutil
import warnings
import torch
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", type=str, default="config/yolov3_default.cfg",
                         help="config file")
    parser.add_argument( "--evaluate", dest="evaluate", action="store_true", 
                         help="Run evaluation on validation set" )
    parser.add_argument( "--weights", type=str, default=None,
                         help="Yolo weights file" )
    parser.add_argument( "--resume-last", dest="resume", action="store_true",
                         help="resume from last stored checkpoint" )
    parser.add_argument( "--resume-from", type=str, default="",
                         help="resume from given checkpoint" )
    parser.add_argument( "--debug", default=False,
                         help="enable debug mode" )

    # model parameters
    parser.add_argument( "--learning-rate", "--lr", default=0.01, type=float,
                         help="learning rate" )
    parser.add_argument( "--momentum", default=0.9, type=float,
                         help="momentum")
    parser.add_argument( "--weight-decay", default=1e-4, type=float,
                         help="weight decay" )

    # training parameters
    parser.add_argument( "--start-epoch", default=0, type=int,
                         help="start epoch number if different from 0" )
    parser.add_argument( "--epochs", default=1, type=int,
                         help="total number of epochs to run" )
    parser.add_argument( "--batch-size", default=128, type=int,
                         help="batch size" )
    parser.add_argument( "--use-cpu", type=int, default=True,
                         help="set True to train on CPU")
    parser.add_argument( "--pretrained", dest="pretrained", action="store_true",
                         help="start from a pretrained model")

    # distributed processing
    parser.add_argument( "--gpu", default=None, type=int, 
                         help="Force training on GPU id" )
    parser.add_argument( "--workers", default=8, type=int,
                         help="number of data loading processes" )
    parser.add_argument( "--nnodes", default=1, type=int, 
                         help="number of nodes for distributed training" )
    parser.add_argument( "--rank", default=0, type=int, 
                         help="node rank for distributed training" )
    parser.add_argument( "--dist-url", default="tcp://10.0.1.164:23456", type=str,
                         help="url used to setup distributed training" )
    parser.add_argument( "--dist-backend", default="nccl", type=str,
                         help="distributed backend" )

    return parser.parse_args()


def parse_config( filename="train.config", config=None ):
    if not os.path.isfile( filename ):
        print( "Config file not found: {}".format( filename ) )
    with open( filename ) as f:
        for line in f:
            key, _, value = line.partition( '=' )
            key = key.strip()
            value = value.strip()
            setattr( config, key, value )


def setup_and_launch( worker_fn=None, config=None ):
    args = parse_args()

    gpus_per_node = torch.cuda.device_count()
    print( "{} GPUs found".format( gpus_per_node ) )
    args.gpus_per_node = gpus_per_node

    torch.manual_seed( 42 )

    config.checkpoint_write = os.path.join( config.checkpoint_path, config.checkpoint_name )

    if args.resume_from:
        args.resume = True
        config.checkpoint_file = os.path.join( config.checkpoint_path, args.resume_from )
    else:
        config.checkpoint_file = os.path.join( config.checkpoint_path, config.checkpoint_name ) 
    
    if not os.path.isfile( config.checkpoint_file ):
        print( "No checkpoint file found: {}".format( config.checkpoint_file ) )
        exit()

    # print the provided config
    print( "Config provided:\n================" )
    config.dump()

    if args.gpu is not None:
        args.world_size = 1
        warnings.warn( "You have chosen to train on a specific GPU")
        worker_fn( args.gpu, 1, args, config )
    else:
        args.world_size = args.nnodes * args.gpus_per_node
        args.batch_size = int( args.batch_size / args.world_size )
        args.workers = int( ( args.workers + args.gpus_per_node - 1 ) / args.gpus_per_node )
        mp.spawn( worker_fn, nprocs=args.gpus_per_node, args=( args, config ) )
    print( "All Done.")


class AverageMeter( object ):
    def __init__( self, name, fmt=":f" ):
        self.name = name
        self.fmt = fmt
        
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update( self, val, n=1 ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__( self ):
        fmt_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt_str.format( **self.__dict__ )

class ProgressMeter( object ):
    def __init__( self, num_batches, meters, prefix='' ):
        self.meters = meters
        self.prefix = prefix
        num_digits = len( str( num_batches // 1 ) )
        fmt = "{:" + str( num_digits ) + "d}"
        self.batch_fmtstr = "[" + fmt + "/" + fmt.format( num_batches ) + "]"

    def display( self, batch ):
        entries = [ self.prefix + self.batch_fmtstr.format( batch ) ]
        entries += [ str( meter ) for meter in self.meters ]
        print( "\t".join( entries ) )

class Config( object ):
    def dump( self ):
        s = ""
        for key, value in self.__dict__.items():
            s = s + "{} = {}\n".format( key, value )
        print( s )

