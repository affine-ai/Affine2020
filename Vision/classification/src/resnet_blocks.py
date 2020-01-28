import torch.nn as nn

activations = nn.ModuleDict( [
    [ "leaky_relu", nn.LeakyReLU() ],
    [ "relu", nn.ReLU() ]
])

class ConvBlock( nn.Module ):
    """
    A conv block with skip connection capability.
    Can be used to implement a ResNet block.
    Default padding is 'same'.
    """
    def __init__( self, inCh, outCh, k, p="same", s=1, activation="relu" ):
        super().__init__()
        self.activationType = activation
        p = ( k - 1 ) // 2 if p is "same" else p
        self.add_module( "conv", nn.Conv2d( in_channels=inCh, \
                                                   out_channels=outCh, \
                                                   kernel_size=k, \
                                                   padding=p, \
                                                   stride=s, \
                                                   bias=False ) )
        self.add_module( "batch_norm", nn.BatchNorm2d( outCh ) )

    def forward( self, x ):
        x = self.conv( x )
        x = self.batch_norm( x )
        if self.activationType is not None:
            x = activations[ self.activationType ]( x )
        return x


class ResNetBlock( nn.Module ):
    def __init__( self, *kargs, **kwargs ):
        super().__init__()
        self.identity = True
        self.skipConnection = None

        try:
            self.activationType = kwargs[ "activation" ]
        except:
            self.activationType = "relu"

        ( blocks, ) = kargs        
        num_blocks = len( blocks )
        
        self.layers = nn.ModuleList()

        for num, block in enumerate( blocks ):
            ( ch, k ) = block
            if num == 0:
                resBlockInputCh = ch
                inputCh = ch
            elif num == num_blocks - 1:
                self.layers.append( ConvBlock( inputCh, ch, k, activation=None ) )
            else:
                self.layers.append( ConvBlock( inputCh, ch, k, activation=self.activationType ) )
                inputCh = ch

        if resBlockInputCh == ch:
            self.identity = True
        else:
            self.identity = False
            self.skipConnection = ConvBlock( resBlockInputCh, ch, 1, activation=None )


    def forward( self, x ):
        if self.identity:
            residual = x
        else:
            residual = self.skipConnection( x )

        for layer in self.layers:
            x = layer( x )

        x = x + residual

        x = activations[ self.activationType ]( x )
        return x

