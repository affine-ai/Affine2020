import torch
import torch.nn as nn
from torchvision.models import resnet50
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#matplotlib.rcParams[ "toolbar" ] = "None"

net = resnet50( pretrained=True ).cpu().eval()
layer = 7
filter = 1016

class SaveParams( object ):
    def __init__( self, module ):
        self.hook = module.register_forward_hook( self.hook_fn )

    def hook_fn( self, module, input, output ):
        self.params = output.clone()
        #print( self.params.size() )

    def remove_hook( self ):
        self.hook.remove()


class ReverseConv( object ):
    def __init__( self, size=64 ):
        global net
        self.size = size
        self.disable_grads( net )

    def visualize( self, layer, filter, lr=0.01 ):

        module = list( net.children() )[ layer ][2].relu
        print( module )
        activations = SaveParams( module )        


        upscale_steps = 15
        steps = 100
        disp_every = 100
        upscale_factor = 1.15
        display_later = True
        blur = 5

        nplots = 1 if display_later else upscale_steps
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.subplots( 1, nplots )


        s = self.size
        img = np.random.rand( s, s, 3 )

        for i in range( upscale_steps ):
            img = img.transpose( 2, 0, 1 )
            img = torch.tensor( [ img ], dtype=torch.float32, requires_grad=True )

            optimizer = torch.optim.Adam( [ img ], lr=lr )

            for j in range( 1, steps + 1 ):
                optimizer.zero_grad()
                bn_img = nn.BatchNorm2d( 3 )( img )
                net( bn_img )
                loss = -activations.params[ 0, filter ].mean()
                loss.backward()
                optimizer.step()

                if ( i * steps + j ) % disp_every == 0:
                    print( "Iter: {}\tloss: {}\tinput: {}".format( i, loss.item(), img.detach().mean() ) )

            s = int( s * upscale_factor )
            img = img.detach().data.numpy()[ 0 ].transpose( 1, 2, 0 )
            img = cv2.resize( img, ( s, s ), interpolation = cv2.INTER_CUBIC )
            if blur:
                img = cv2.blur( img, ( blur, blur ) )

            if not display_later and nplots > 1:
                ax[ i ].imshow( img )
        
        if display_later:
            ax.imshow( img )           

        fig.tight_layout()
        fig.show()

        activations.remove_hook()
        torch.save( { "image": img, "optimizer": optimizer.state_dict() }, "reverse.chkpoint.tar" )
        input( "Press Enter to end..." )
        
    def show_img( self, image ):
        img = image.squeeze( 0 ).permute( 1, 2, 0 )
        plt.imshow( img.detach() )
        plt.show( block=False )

    def disable_grads( self, model ):
        for param in model.parameters():
            param.requires_grad = False


reverse = ReverseConv()
reverse.visualize( layer, filter )

