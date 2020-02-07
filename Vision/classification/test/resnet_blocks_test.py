#!/usr/bin/env python3

from Affine.Vision.classification.src.resnet_blocks import ResnetBlock
import torch

def basic_test():
    torch.manual_seed( 1 )
    resnet = ResnetBlock( in_channels=3, F1=3, F2=6, F3=3, kernel=3, activation_type="relu" )

    x = torch.rand( ( 1, 3, 32, 32 ) )
    x = resnet( x )
    print( x.size() )

if __name__ == "__main__":
    basic_test()