import torch.nn as nn
from collections import OrderedDict

activations = nn.ModuleDict( [
    [ "leaky_relu", nn.LeakyReLU( inplace=True ) ],
    [ "relu", nn.ReLU( inplace=True ) ],
    [ "none", nn.Identity() ], ] )

def conv_unit( in_channels, out_channels, kernel, stride=1, padding="same", activation_type="relu" ):
    if padding == "same":
        padding = int( ( kernel - 1 ) / 2 )
    else:
        padding = 0
    return nn.Sequential( nn.Conv2d( in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel, 
                                        stride=stride, 
                                        padding=padding,
                                        bias=False ),
                            nn.BatchNorm2d( num_features=out_channels ),
                            activations[ activation_type ] )

class ResnetBottleneck( nn.Module ):
    def __init__( self, in_channels, F1, F2, F3, kernel=3, activation_type="relu" ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = F3
        self.identity = True
        self.skip_conn = None
        self.activation_type = activation_type

        self.layers = nn.ModuleList()

        self.layers.append( conv_unit( in_channels, F1, 1, padding="valid" ) )
        self.layers.append( conv_unit( F1, F2, kernel, padding="same" ) )
        self.layers.append( conv_unit( F2, F3, 1, padding="valid", activation_type="none" ) )

    def forward( self, x ):
        if self.in_channels == self.out_channels:
            residual = x
        else:
            residual = conv_unit( x, self.out_channels, 1, activation_type="none" )

        for layer in self.layers:
            x = layer( x )
        x = x + residual
        return activations[ self.activation_type ]( x )


class ResnetBasic( nn.Module ):
    def __init__( self, channels, kernel=3, activation_type="relu" ):
        super().__init__()

        self.activation_type = activation_type

        padding = int( ( kernel - 1 ) / 2 )

        self.conv1 = nn.Conv2d( in_channels=channels, 
                                out_channels=channels, 
                                kernel_size=kernel, 
                                stride=1, 
                                padding=padding,
                                bias=False )
        self.bn1 = nn.BatchNorm2d( num_features=channels )

        self.conv2 = nn.Conv2d( in_channels=channels, 
                                out_channels=channels, 
                                kernel_size=kernel, 
                                stride=1, 
                                padding=padding,
                                bias=False )
        self.bn2 = nn.BatchNorm2d( num_features=channels )
        self.relu = activations[ activation_type ]

    def forward( self, x ):
        residual = x
        x = self.conv1( x )
        x = self.bn1( x )
        x = self.relu( x )
        x = self.conv2( x )
        x = self.bn2( x )

        x = x + residual
        return self.relu( x )
    