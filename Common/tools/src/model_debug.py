from Affine.Common.utils.src.train_utils import Config
from Affine.Vision.classification.src.darknet53 import Darknet53

import os
import cmd
import pdb
import sys
import traceback
import code
import collections
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models import *
from PIL import Image

model = None
image = None
out = None


class Shell( cmd.Cmd ):
    def __init__( self, model, config ):
        super().__init__()
        self.model = model
        self.config = config
        self.image_size = 224
        self.rc_lines = []

        self.curframe = sys._getframe().f_back

        try:
            with open( ".modeldebugrc" ) as rc_file:
                self.rc_lines.extend( rc_file )
        except OSError:
            pass        
        self.exec_rc()

    # Functions overridden from base class go here
    def precmd( self, line ):
        return line

    def onecmd( self, line ):
        if line.find( " " ) > 0:
            args = line.split( " " )
            map( str.strip, args )
            c = "{}_{}".format( args.pop( 0 ), args.pop( 0 ) )
            if ( callable( getattr( self, "do_" + c, None ) ) ):
                line = "{} {}".format( c, " ".join( args ) )
                print( "Executing {}".format( line ) )
        cmd.Cmd.onecmd( self, line )

    def default( self, line ):
        if line[ :1 ] == '!':
            line  = line[ 1: ]
        locals = self.curframe.f_locals
        globals = self.curframe.f_globals
        try:
            code = compile( line + "\n", "<stdin>", "single" )
            saved_stdin = sys.stdin
            saved_stdout = sys.stdout
            try:
                sys.stdin = self.stdin
                sys.stdout = self.stdout
                exec( code, globals, locals )
            finally:
                sys.stdin = saved_stdin
                sys.stdout = saved_stdout
        except:
            exec_info = sys.exc_info()[ :2 ]
            self.error( traceback.format_exception_only( *exec_info )[ -1 ].strip() )


    # All do_* command functions go here

    def do_quit( self, args ):
        """Exits the shell"""
        print( "Exiting shell" )
        raise SystemExit

    def do_summary( self, args ):
        """Prints pytorch model summary"""
        try:
            for layer in model.layers:
                print( "{}  {}".format( layer._get_name(), layer.size() ) )
        except:
            self.error( sys.exc_info()[ 1 ] )

    def do_load_image( self, args ):
        """load a single image"""
        global image
        
        image_path = os.path.join( self.config.image_path, args )
        if not os.path.isfile( image_path ):
            print( "Image not found")
            return
        print( "Loading image {}".format( image_path ) )
        image = Image.open( image_path )
        transform = transforms.Compose( [ transforms.Resize( ( self.image_size, self.image_size ) ),
                                            transforms.ToTensor() ] )
        image = transform( image ).float()
        image = image.view( 1, *image.shape )

    def do_load_checkpoint( self, args ):
        """Load a checkpoint file into the model.
        If no file is specified, location specified in the
        config file is used"""
        global model
        
        if args:
            file = os.path.join( self.config.checkpoint_path, args )
        else:
            file = os.path.join( self.config.checkpoint_path, self.config.checkpoint_name )
        if not os.path.isfile( file ):
            self.error( "Checkpoint file not found" )
            return

        chkpoint = torch.load( file, map_location="cpu", )
        self.message( "Loading checkpoint file: {}".format( file ) )
        
        state_dict = chkpoint[ "model_state_dict" ]

        try:
            model.load_state_dict( state_dict )
        except RuntimeError:
            new_state_dict = collections.OrderedDict( [ ( k[ 7: ], v ) for k,v in state_dict.items() 
                                                                        if k.startswith( "module" ) ] )
            model.load_state_dict( new_state_dict )

    do_load_chkp = do_load_checkpoint

    def do_show_image( self, args ):
        plt.imshow( image[ 0 ].permute( 1, 2, 0 ) )
        plt.show( block=False )
    
    def do_show_firstconv( self, args ):
        if args:
            try:
                net = self.curframe.f_globals[ args ]
            except:
                self.error( "Could not find specified model {}".format( args ) )
                return
        else:
            net = model
        try:
            w = net.conv1.weight.detach()
        except AttributeError:
            w = net.layers[0][0].weight.detach()

        w = w.permute( 0, 2, 3, 1 )
        nfilters = w.size()[ 0 ]
        s = int( np.floor( np.sqrt( nfilters ) ) )
        
        # if number of filters is not a perfect square, we pad the tensor so we can 
        # display it in a square grid
        if s ** 2 != nfilters:
            s += 1
            npad = s ** 2 - nfilters
            w = torch.cat( ( w, torch.ones( ( npad, *w.size()[ 1: ] ) ) ), dim=0 )

        grid_w = torch.cat( tuple( torch.cat( tuple( w[ k ] for k in range( j * s, j * s + s ) ), dim=1 ) 
                                                               for j in range( s ) ), dim=0 )
        plt.imshow( grid_w )
        plt.show( block=False )

    do_show_fconv = do_show_firstconv


    # All local functions specific to this class go here
    def error( self, err_msg ):
        print( "***", err_msg, file=self.stdout )

    def message( self, msg ):
        print( msg, file=self.stdout )

    def exec_rc( self ):
        if not self.rc_lines:
            return

        print( "\nExecuting rc file" )
        num = 1
        while self.rc_lines:
            line = self.rc_lines.pop( 0 ).strip()
            print( "{}: {}".format( num, line ), end='' )
            num += 1
            if not line or "#" in line[ 0 ]:
                print()
                continue
            self.onecmd( line )
            print( " ...Done" )
        print()

    def _cmdloop( self, intro_header ):
        while True:
            try:
                self.allow_kbdint = True
                self.cmdloop( intro_header )
                self.allow_kbdint = False
                break
            except KeyboardInterrupt:
                self.message( "**Keyboard Interrupt" )
            except RuntimeError as e:
                self.error( e )


if __name__ == "__main__":
    config = Config()
    config.checkpoint_path = "/home/vipul/Affine/Vision/classification/train/checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    config.image_path = "/home/vipul/Affine/Vision/classification/test/images/"

    shell = Shell( model, config )
    shell.prompt = '> '
    shell._cmdloop( "Welcome to the shell" )