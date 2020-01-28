from Affine.Vision.classification.src.resnet_blocks import ConvBlock, ResNetBlock
import torch.nn as nn

class Darknet53( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append( ConvBlock( 3, 32, 3 ) )
        self.layers.append( ConvBlock( 32, 64, 3, s=2 ) )
        self.layers.append( ResNetBlock( [ ( 64, 0 ), 
                                            ( 32, 1 ), 
                                            ( 64, 3 ) ] ) )
        self.layers.append( ConvBlock( 64, 128, 3, s=2 ) )
        for _ in range( 2 ):
            self.layers.append( ResNetBlock( [ ( 128, 0 ), 
                                                ( 64, 1 ), 
                                                ( 128, 3 ) ] ) )
        self.layers.append( ConvBlock( 128, 256, 3, s=2 ) )
        for _ in range( 8 ):
            self.layers.append( ResNetBlock( [ ( 256, 0 ),
                                                ( 128, 1 ), 
                                                ( 256, 3 ) ] ) )
        self.layers.append( ConvBlock( 256, 512, 3, s=2 ) )
        for _ in range( 8 ):
            self.layers.append( ResNetBlock( [ ( 512, 0 ),
                                                ( 256, 1 ), 
                                                ( 512, 3 ) ] ) )
        self.layers.append( ConvBlock( 512, 1024, 3, s=2 ) )
        for _ in range( 4 ):
            self.layers.append( ResNetBlock( [ ( 1024, 0 ),
                                                ( 512, 1 ), 
                                                ( 1024, 3 ) ] ) )

    def forward( self, x ):
        for layer in self.layers:
            x = layer( x )
        return x
