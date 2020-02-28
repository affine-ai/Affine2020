import os
import argparse
import random
import scipy.io
import shutil
import warnings
import torch
import torch.multiprocessing as mp
import numpy as np
from collections import OrderedDict
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", type=str, default="config/train.cfg",
                         help="config file")
    parser.add_argument( "--evaluate", dest="evaluate", action="store_true", 
                         help="Run evaluation on validation set" )
    parser.add_argument( "--weights", type=str, default=None,
                         help="Yolo weights file" )
    parser.add_argument( "--resume-last", dest="resume", action="store_true",
                         help="resume from last stored checkpoint" )
    parser.add_argument( "--resume-from", type=str, default="",
                         help="resume from given checkpoint" )

    # optimizer parameters
    parser.add_argument( "--momentum", default=0.9, type=float,
                         help="momentum")
    parser.add_argument( "--weight-decay", default=1e-6, type=float,
                         help="weight decay" )
    parser.add_argument( "--base-lr", default=0.0001, type=float,
                         help="min learning rate" )
    parser.add_argument( "--max-lr", default=0.1, type=float,
                         help="max learning rate" )
    parser.add_argument( "--stepsize", default=1000, type=int,
                         help="half the number of iterations to cycle the learning rate" )
    parser.add_argument( "--lr-policy", default="triangle", type=str,
                         help="Select the learning rate adjustment policy" )    
    
    # training parameters
    parser.add_argument( "--start-epoch", default=1, type=int,
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

    # debugging and profiling
    parser.add_argument( "--debug", default=False,
                         help="enable debug mode" )
    parser.add_argument( "--prof", default=0, type=int,
                         help="enable profiling" )
    return parser.parse_args()


def parse_config( filename ):
    config = Config()
    if not os.path.isfile( filename ):
        print( "Config file not found: {}".format( filename ) )
        return config

    lines = []
    try:
        with open( filename ) as f:
            lines.extend( f )
    except OSError:
        print( "Could not load config file" )
        return config

    if not lines:
        print( "Config file is empty" )
        return config

    num = 1
    while lines:
        line = lines.pop( 0 ).strip()
        num += 1
        if not line or "#" in line[ 0 ]:
            continue
        if line.find( "=" ) > 0:
            args = line.split( "=" )
            var, val = args.pop( 0 ).strip(), args.pop( 0 ).strip()
            setattr( config, var, val )
    return config


def setup_and_launch( worker_fn=None, config=None ):
    """Pre-process args and launch the entry function into training
    """
    args = parse_args()

    gpus_per_node = torch.cuda.device_count()
    print( "Found {} GPUs".format( gpus_per_node ) )
    args.gpus_per_node = gpus_per_node

    np.random.seed( 42 )
    torch.manual_seed( 42 )
    torch.cuda.manual_seed( 42 )

    if config is None:
        config = parse_config( args.config )

    # print the provided config
    print( "==========Config provided==========" )
    print( config )
    print( "===================================\n" )

    config.checkpoint_write = os.path.join( config.checkpoint_path, config.checkpoint_name )
    if args.resume_from:
        args.resume = True
        config.checkpoint_file = os.path.join( config.checkpoint_path, args.resume_from )
    else:
        config.checkpoint_file = os.path.join( config.checkpoint_path, config.checkpoint_name ) 
    
    if args.resume:
        if not os.path.isfile( config.checkpoint_file ):
            print( "No checkpoint file found: {}".format( config.checkpoint_file ) )
            exit()
        else:
            print( "\n***You have chosen to resume from a checkpoint\n***\n" )
    
    distributed = args.gpu is None

    if distributed:
        args.world_size = args.nnodes * args.gpus_per_node
        args.batch_size = int( args.batch_size / args.world_size )
        args.workers = int( ( args.workers + args.gpus_per_node - 1 ) / args.gpus_per_node )
        mp.spawn( worker_fn, nprocs=args.world_size, args=( args, config ) )
    else:
        args.world_size = 1
        warnings.warn( "You have chosen to train on a specific GPU")
        worker_fn( args.gpu, args, config )

    print( "All Done.")


def load_checkpoint( model, checkpoint_path ):
    """Loads the model state from a checkpoint file
    Inputs:
        model: reference to the model
        checkpoint_path: full path to the checkpoint file
    Returns: 
        True if successfully loaded the checkpoint otherwise False
    """
    if not os.path.isfile( checkpoint_path ):
        print( "Checkpoint file not found" )
        return False

    print( "Loading checkpoint {}".format( checkpoint_path ) )
    checkpoint = torch.load( checkpoint_path, map_location='cpu' )

    state_dict = checkpoint[ "model" ]
    attempt = 0
    while attempt < 2:
        try:
            model.load_state_dict( state_dict )
        except RuntimeError:
            attempt += 1
            state_dict = OrderedDict( [ ( k[ 7: ], v ) for k, v in state_dict.items() 
                                                                if k.startswith( "module" ) ] )
        else:
            return True
    return False


def adjust_learning_rate( optimizer, i, args, policy ):
    """learning rate schedule
    """
    cycle = math.floor( 1 + i / ( 2 * args.stepsize ) )
    if policy is "triangle2":
        range = ( args.max_lr - args.base_lr ) / pow( 2, int( cycle - 1 ) )
    else:
        range = ( args.max_lr - args.base_lr )

    x = abs( i / args.stepsize - 2 * cycle + 1 )
    lr = args.base_lr + range * max( 0.0, ( 1.0 - x ) )

    for param_group in optimizer.param_groups:
        param_group[ 'lr' ] = lr
    return lr


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
        # If self.avg is a NaN, reset it
        if self.avg != self.avg:
            self.avg = 0
            self.count = 0

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
    def __str__( self ):
        s = ""
        for key, value in self.__dict__.items():
            s = s + "{} = {}\n".format( key, value )
            s = s.rstrip( "\n" )
        return s
