from Affine.Common.utils.src.train_utils import Config
from Affine.Vision.classification.src.darknet53 import Darknet53

import os
import cmd
import pdb
import sys
import traceback
import code

import numpy as np
import torch
import torchvision
from torchvision import transforms
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
        self.curframe = sys._getframe().f_back

    # Functions overriddein from base class go here
    def precmd( self, line ):
        return line

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
    def do_load_model( self, args ):
        """Load model from given checkpoint file. \
        If no file is specified, default checkpoint file is used"""
        config = self.config
        config.checkpoint_file = os.path.join( config.checkpoint_path, config.checkpoint_name ) 

        if os.path.isfile( config.checkpoint_file ):
            print( "Loading model from: {}".format( config.checkpoint_file ) )
            checkpoint = torch.load( config.checkpoint_file, map_location='cpu' )
            self.model.load_state_dict( checkpoint[ "model_state_dict" ], strict=False )

    def do_quit( self, args ):
        """Exits the shell"""
        print( "Exiting shell" )
        raise SystemExit

    def do_summary( self, args ):
        """Prints pytorch model summary"""
        print( self.model )

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
        
        file = os.path.join( self.config.checkpoint_path, self.config.checkpoint_name )
        if not os.path.isfile( file ):
            self.error( "Checkpoint file not found" )
            return

        chkpoint = torch.load( file, map_location="cpu", )
        self.message( "Loading checkpoint file: {}".format( file ) )
        try:
            model.load_state_dict( chkpoint[ "model_state_dict" ], strict=False )
        except:
            self.error( sys.exc_info()[ 0 ] )


    # All local functions specific to this class go here
    def error( self, err_msg ):
        print( "**", err_msg, file=self.stdout )

    def message( self, msg ):
        print( msg, file=self.stdout )

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
                self.message( e )


if __name__ == "__main__":
    config = Config()
    config.checkpoint_path = "/home/vipul/Affine/Vision/classification/train/checkpoint"
    config.checkpoint_name = "checkpoint.pth.tar"
    config.image_path = "/home/vipul/Affine/Vision/classification/test/images/"

    shell = Shell( model, config )
    shell.prompt = '> '
    shell._cmdloop( "Welcome to the shell" )