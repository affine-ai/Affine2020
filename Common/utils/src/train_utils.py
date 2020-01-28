import os
import argparse
import random
import scipy.io
import shutil

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


def parse_config( config, filename="train.config" ):
    if not os.path.isfile( filename ):
        print( "Config file not found: {}".format( filename ) )
    with open( filename ) as f:
        for line in f:
            key, _, value = line.partition( '=' )
            key = key.strip()
            value = value.strip()
            config[ key ] = value
