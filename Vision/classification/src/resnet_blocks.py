import torch.nn as nn

activations = nn.ModuleDict( [
    [ "leaky_relu", nn.LeakyReLU() ],
    [ "relu", nn.ReLU() ],
    [ "none", nn.Identity() ], ] )

class ResnetBlock( nn.Module ):
    def __init__( self, in_channels, F1, F2, F3, kernel, activation_type="relu" ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = F3
        self.identity = True
        self.skip_conn = None
        self.activation_type = activation_type

        self.layers = nn.ModuleList()

        self.layers.append( self.conv_unit( in_channels, F1, 1, ) )
        self.layers.append( self.conv_unit( F1, F2, kernel ) )
        self.layers.append( self.conv_unit( F2, F3, 1, activation_type="none" ) )

    def conv_unit( self, in_channels, out_channels, kernel, stride=1, padding=0, activation_type="relu" ):
        return nn.Sequential( nn.Conv2d( in_channels=in_channels, 
                                           out_channels=out_channels, 
                                           kernel_size=kernel, 
                                           stride=stride, 
                                           padding=padding ),
                                nn.BatchNorm2d( num_features=out_channels ),
                                activations[ activation_type ] )

    def forward( self, x ):
        if self.in_channels == self.out_channels:
            residual = x
        else:
            residual = self.conv_unit( x, self.out_channels, 1, activation_type="none" )

        for layer in self.layers:
            x = layer( x )
        x = x + residual
        return activations[ self.activation_type ]( x )

