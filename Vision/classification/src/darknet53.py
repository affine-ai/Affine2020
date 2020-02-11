from Affine.Vision.classification.src.resnet_blocks import ResnetBottleneck, ResnetBasic, conv_unit
import torch.nn as nn
import torch


class Darknet53( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append( conv_unit( 3, 32, 7 ) )
        self.layers.append( conv_unit( 32, 64, 3, stride=2 ) )
        self.layers.append( ResnetBottleneck( 64, 64, 32, 64 ) )
        self.layers.append( conv_unit( 64, 128, 3, stride=2 ) )
        for _ in range( 2 ):
            self.layers.append( ResnetBottleneck( 128, 128, 64, 128 ) )
        self.layers.append( conv_unit( 128, 256, 3, stride=2 ) )
        for _ in range( 8 ):
            self.layers.append( ResnetBottleneck( 256, 256, 128, 256 ) )
        self.layers.append( conv_unit( 256, 512, 3, stride=2 ) )
        for _ in range( 8 ):
            self.layers.append( ResnetBottleneck( 512, 512, 256, 512 ) )
        self.layers.append( conv_unit( 512, 1024, 3, stride=2 ) )
        for _ in range( 4 ):
            self.layers.append( ResnetBottleneck( 1024, 1024, 512, 1024 ) )
        self.fc = nn.Linear( 7 * 7 * 1024, 1000 )

    def forward( self, x ):
        for layer in self.layers:
            x = layer( x )
        x = torch.flatten( x, 1 )
        x = self.fc( x )
        return x


class darknet( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.blocks = nn.Sequential( conv_unit( 3, 32, 7 ),
                                     conv_unit( 32, 64, 3, stride=2 ),
                                     ResnetBasic( 64 ),
                                     conv_unit( 64, 128, 3, stride=2 ),
                                     *[ ResnetBasic( 128 ) for _ in range( 2 ) ],
                                     conv_unit( 128, 256, 3, stride=2 ),
                                     *[ ResnetBasic( 256 ) for _ in range( 8 ) ],
                                     conv_unit( 256, 512, 3, stride=2 ),
                                     *[ ResnetBasic( 512 ) for _ in range( 8 ) ],
                                     conv_unit( 512, 1024, 3, stride=2 ),
                                     *[ ResnetBasic( 1024 ) for _ in range( 4 ) ]
                                     )

        self.fc = nn.Linear( 7 * 7 * 1024, 1000 )

    def forward( self, x ):
        x = self.blocks( x )
        x = torch.flatten( x, 1 )
        x = self.fc( x )
        return x
