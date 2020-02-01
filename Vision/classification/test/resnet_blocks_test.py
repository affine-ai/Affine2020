from Affine.Vision.classification.src.resnet_blocks import ResnetBlock
import torch


torch.manual_seed( 1 )
resnet = ResnetBlock( in_channels=3, F1=3, F2=6, F3=3, kernel=3, activation_type="relu" )

for module in resnet.modules():
    x = torch.rand( ( 32, 32, 3 ) )
    print( module )
