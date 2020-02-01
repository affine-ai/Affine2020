from Affine.Vision.classification.src.resnet_blocks import ResnetBlock
import torch


resnet = ResnetBlock( [ ( 8, 0 ),
                        ( 16, 3 ),
                        ( 8, 1 ) ] )

for module in resnet.modules():
    print( module )
