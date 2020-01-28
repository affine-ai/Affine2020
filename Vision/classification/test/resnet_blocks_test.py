from YOLO.src.resnet_blocks import *
import torch


resnet = ResNetBlock( [ ( 8, 0 ),
                        ( 16, 3 ),
                        ( 8, 1 ) ] )

for module in resnet.modules():
    print( module )
