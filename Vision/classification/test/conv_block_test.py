from YOLO.src.resnet_blocks import *
import torch

in_channels = 3
out_channels = 1
kernel = 3
pad = "same"
stride = 1
activation = None

conv = ConvBlock( in_channels, \
                  out_channels, 
                  kernel, \
                  pad, \
                  stride, \
                  activation )
 

#============Test 2 ====================
#Change activation to relu and verify
#=======================================

activation = "relu"

conv = ConvBlock( in_channels, \
                  out_channels, 
                  kernel, \
                  pad, \
                  stride, \
                  activation )

