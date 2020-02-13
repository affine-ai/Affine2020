#! /usr/bin/env python3
from Affine.Vision.classification.src.darknet53 import Darknet53, darknet

import os, sys, code, traceback
import cmd, readline
import atexit
import collections
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torchvision.models import *
from PIL import Image


model = None
image = None
out = None

class Config( object ):
    pass 

config = Config()

class LayerMeta( object ):
    def __init__( self, module ):
        self.out = None
        self.fn = None
        self.hook = module.register_forward_hook( self.hook_fn )
        print( "Registering forward hook" )

    def hook_fn( self, module, input, output ):
        self.out = output.clone().detach()

    def available( self ):
        if self.out is not None:
            return True
        return False

    def data( self ):
        if self.fn:
            try:
                return self.fn( self.out )
            except:
                return None
        else:
            return self.out
    
    def size( self ):
        return self.out.size()

    def dim( self ):
        return self.out.dim()

    def mean( self ):
        return self.out.mean()

    def modify( self, fn ):
        self.fn = fn

    def close( self ):
        self.hook.remove()
        print( "Removing hooks" )

class Shell( cmd.Cmd ):
    def __init__( self, config ):
        super().__init__()
        self.config = config
        self.image_size = 224
        self.rc_lines = []
        self.device = "cpu"
        self.cur_layer = None
        self.output_hooked = None
        self.cur_frame = sys._getframe().f_back

        try:
            with open( ".pmdebugrc" ) as rc_file:
                self.rc_lines.extend( rc_file )
        except OSError:
            pass        
        self.exec_rc()

        self.init_history( histfile=".pmdebug_history" )
        atexit.register( self.save_history, histfile=".pmdebug_history" )

    ##############################################
    # Functions overridden from base class go here
    ##############################################
    def precmd( self, line ):
        return line

    def onecmd( self, line ):
        if line.find( " " ) > 0:
            args = line.split( " " )
            map( str.strip, args )
            c = "{}_{}".format( args.pop( 0 ), args.pop( 0 ) )
            if ( callable( getattr( self, "do_" + c, None ) ) ):
                line = "{} {}".format( c, " ".join( args ) )
        cmd.Cmd.onecmd( self, line )

    def default( self, line ):
        if line[ :1 ] == '!':
            line  = line[ 1: ]
        locals = self.cur_frame.f_locals
        globals = self.cur_frame.f_globals
        try:
            code = compile( line + "\n", "<stdin>", "single" )
            saved_stdin = sys.stdin
            saved_stdout = sys.stdout
            sys.stdin = self.stdin
            sys.stdout = self.stdout

            try:
                exec( code, globals, locals )
            finally:
                sys.stdin = saved_stdin
                sys.stdout = saved_stdout
        except:
            exec_info = sys.exc_info()[ :2 ]
            self.error( traceback.format_exception_only( *exec_info )[ -1 ].strip() )


    ####################################
    # All do_* command functions go here
    ####################################
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

        chkpoint = torch.load( file, map_location="cpu" )
        self.message( "Loading checkpoint file: {}".format( file ) )
        
        state_dict = chkpoint[ "model" ]

        try:
            model.load_state_dict( state_dict )
        except RuntimeError:
            new_state_dict = collections.OrderedDict( [ ( k[ 7: ], v ) for k, v in state_dict.items() 
                                                                        if k.startswith( "module" ) ] )
            model.load_state_dict( new_state_dict )

    do_load_chkp = do_load_checkpoint


    def do_show_image( self, args ):
        img = self.load_from_context( args, default=image )
        if img is None:
            self.error( "Could not find image" )
            return

        if img.size( 0 ) == 1:
            img = img.squeeze( 0 )
        plt.imshow( img.permute( 1, 2, 0 ) )
        plt.show( block=False )
    
    do_show_img = do_show_image


    def do_show_firstconv( self, args ):
        net = self.load_from_context( args, default=model )
        if net is None:
            self.error( "Could not find specified model {}".format( args ) )
            return

        conv = self.find_first_conv( net )
        if not conv:
            self.error( "No Conv2d layer found" )
            return

        w = conv.weight.detach()
        w = w.permute( 0, 2, 3, 1 )

        nfilters = w.size()[ 0 ]
        s = int( np.floor( np.sqrt( nfilters ) ) )
        
        # if the number of filters is not a perfect square, we pad 
        # the tensor so that we can display it in a square grid
        if pow( s, 2 ) < nfilters:
            s += 1
            npad = pow( s, 2, nfilters )
            w = torch.cat( ( w, torch.ones( ( npad, *w.size()[ 1: ] ) ) ), dim=0 )

        grid_w = torch.cat( tuple( torch.cat( tuple( w[ k ] for k in range( j * s, j * s + s ) ), dim=1 ) 
                                                               for j in range( s ) ), dim=0 )
        plt.imshow( grid_w )
        plt.show( block=False )

    do_show_fconv = do_show_firstconv


    def do_grab_output( self, args ):
        layer = self.load_from_context( args, self.cur_layer )
        if layer == None:
            print( "No feasible layer found")
            return
        self.output_hooked = LayerMeta( layer )
        
    do_grab_out = do_grab_output
    do_grab_hook = do_grab_output


    def do_release_output( self, args ):
        self.output_hooked.close()
        self.output_hooked = None

    do_rel_out = do_release_output
    do_rel_hook = do_release_output


    def do_plot_hook( self, args ):
        if self.output_hooked == None or self.output_hooked.available() == False:
            print( "No hooked outputs available")
            return
    
        if self.output_hooked.dim() == 4:
            data = self.output_hooked.data().squeeze( 0 )
            s0 = data.size( 0 )
            index = np.arange( s0 )
            y_data = [ torch.mean( data[ t ] ).float().item() for t in range( s0 ) ]
            top5 = self.top_n( 5, y_data )
            plt.bar( index, y_data, align="center", width=2 )
            for i, v in top5:
                plt.text( i, v, "{}".format( i ) )
            plt.title( "Histogram of hooked data" )
            plt.grid()
            plt.show( block=False )

    def do_modify_hook( self, args ):
        if args == "relu":
            fn = torch.nn.ReLU( inplace=True )
        elif args == "mean":
            fn = torch.mean
        elif args == "none" or args == "None":
            fn = None
        else:
            fn = self.load_from_context( args, default=None )
            if not fn:
                print( "Could not find function {}".format( args ) )
                return
        if self.output_hooked == None:
            print( "No hooks available" )
        
        self.output_hooked.modify( fn )

    do_mod_hook = do_modify_hook


    ###########################
    # Utility functions go here
    ###########################
    def find_first_conv( self, net ):
        for layer in net.children():
            if isinstance( layer, nn.ModuleList ) or isinstance( layer, nn.Sequential ):
                layer = self.find_first_conv( layer )
            if isinstance( layer, nn.Conv2d ):
                return layer
        return None

    def load_from_context( self, name, default=None ):
        if not name:
            return default

        if name in self.cur_frame.f_globals:
            return self.cur_frame.f_globals[ name ]
        else:
            return default

    def top_n( self, n, ar ):
        index = np.argpartition( ar, -n )[ -n: ]
        return [ ( i, ar[ i ] ) for i in index ]

    ####################################################
    # Helper functions to debugger functionality go here
    ####################################################
    def error( self, err_msg ):
        print( "***.{}".format( err_msg ), file=self.stdout )

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

    def init_history( self, histfile ):
        try:
            readline.read_history_file( histfile )
        except FileNotFoundError:
            pass        
        readline.set_history_length( 2000 )
        readline.set_auto_history( True )

    def save_history( self, histfile ):
        print( "Saving history" )
        readline.write_history_file( histfile )

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
    shell = Shell( config )
    shell.prompt = '>> '
    shell._cmdloop( "Welcome to the shell" )