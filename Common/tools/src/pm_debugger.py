#! /usr/bin/env python3
from Affine.Vision.classification.src.darknet53 import Darknet53, darknet

import os, sys, code, traceback
import cmd, readline
import atexit
from collections import OrderedDict
import curses
from curses import wrapper

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

# Disable the top menubar on plots
matplotlib.rcParams[ "toolbar" ] = "None"

model = None
image = None
out = None

class Config( object ):
    pass 

config = Config()


class GraphWindow( object ):
    def __init__( self ):
        self.fig = plt.figure()

    def reset_window( self ):
        for ax in self.fig.axes:
            self.fig.delaxes( ax )

    def current_axes( self ):
        self.reset_window()
        return self.fig.subplots( 1, 1 )

    def imshow( self, image ):
        ax = self.current_axes()
        ax.imshow( image )
        self.show_graph()

    def show_graph( self ):
        self.fig.canvas.draw()
        self.fig.show()

    def close( self ):
        plt.close()

class ModelMeta( object ):
    def __init__( self, model ):
        self.model = model
        self.cur_layer = None
        self.layers = OrderedDict()

    def init_layer( self ):
        id, layer = self.find_last_instance( self.model, layer=nn.ReLU )
        layer_info = LayerMeta( layer, id )
        self.layers[ tuple( id ) ] = layer_info
        self.cur_layer = layer_info
        return layer_info

    def get_cur_id_layer( self ):
        if not self.cur_layer:
            self.init_layer()
        return self.cur_layer.id, self.cur_layer.layer

    def get_cur_layer_info( self ):
        if not self.cur_layer:
            self.init_layer()
        return self.cur_layer

    def up( self ):
        return self.traverse_updown( dir=-1 )

    def down( self ):
        return self.traverse_updown( dir=1 )

    def traverse_updown( self, dir ):
        id = self.get_cur_layer_info().id
        new_id, new_layer = self.find_instance_by_id( self.model, id, dir=dir )
        
        if not new_id:
            return False

        id, layer = new_id, new_layer

        if tuple( id ) in self.layers:
            self.cur_layer = self.layers[ tuple( id ) ]
        else:
            self.cur_layer = LayerMeta( layer, id )
            self.layers[ tuple( id ) ] = self.cur_layer
        return True

    def find_instance_by_id( self, net, key, dir ):
        cur_frame = []
        frame = []
        leaf = None
        terminate = False
        terminate_next = False

        def _recurse_layer( layer ):
            nonlocal key
            nonlocal cur_frame
            nonlocal frame
            nonlocal leaf
            nonlocal terminate
            nonlocal terminate_next

            for i, m in enumerate( layer.children() ):
                if terminate:
                    break

                cur_frame.append( i )

                if cur_frame == key:
                    if dir == -1:
                        terminate = True
                    elif dir == 1:
                        terminate_next = True
                        frame = []
                    elif dir == 0:
                        terminate == True
                        frame, leaf = cur_frame.copy(), m
                else:
                    # We don't have a key match, treat it like a normal iteration
                    # If this is a leaf node, save it's location else recurse further
                    if not list( m.children() ):
                        frame, leaf = cur_frame.copy(), m
                        terminate = terminate_next
                    else:
                        _recurse_layer( m )

                cur_frame.pop()       
            return

        _recurse_layer( net )
        return frame, leaf


    def find_last_instance( self, net, layer=nn.Conv2d, cur_frame=[], found_frame=[] ):
        """This method does a depth first search and finds the last instance of the
        specifiied layer in the tree"""
        found = None
        ret_found = None

        for i, l in enumerate( net.children() ):
            cur_frame.append( i )

            if isinstance( l, layer ):
                found = l
                if cur_frame > found_frame:
                    found_frame = cur_frame.copy()

            found_frame, ret_found = self.find_last_instance( l, layer=layer,
                                                            cur_frame=cur_frame, 
                                                            found_frame=found_frame )
            if isinstance( ret_found, layer ):
                found = ret_found
            cur_frame.pop()
        return found_frame, found


class LayerMeta( object ):
    def __init__( self, layer, id=[] ):
        self.out = None
        self.post_process_fn = None
        self.layer = layer
        self.id = id

    def register_forward_hook( self, hook_fn=None ):
        if hook_fn is None:
            hook_fn = self.fhook_fn
        self.fhook = self.layer.register_forward_hook( hook_fn )

    def fhook_fn( self, layer, input, output ):
        self.out = output.clone().detach()

    def available( self ):
        if self.out is not None:
            return True
        return False

    def data( self, raw=False ):
        if self.post_process_fn and raw is False:
            try:
                return self.post_process_fn( self.out )
            except:
                return None
        else:
            return self.out
    
    def size( self, dim=None ):
        if dim is None:
            return self.out.size()
        else:
            return self.out.size( dim )

    def dim( self ):
        return self.out.dim()

    def post_process_hook( self, fn ):
        if callable( fn ):
            self.post_process_fn = fn
        else:
            return False

    def has_post_process( self ):
        if self.post_process_fn:
            return True
        return False

    def close( self ):
        self.fhook.remove()


class Shell( cmd.Cmd ):
    def __init__( self, config ):
        super().__init__()
        self.config = config
        self.image_size = 224
        self.rc_lines = []
        self.device = "cpu"
        self.models = {}
        self.cur_model = None
        self.data_post_process_fn = None
        self.cur_frame = sys._getframe().f_back

        self.fig = GraphWindow()

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

        is_assign = False
        if line.find( '=' ) > 0:
            var, _ = line.split( '=', maxsplit=1 )
            var = var.strip()
            is_assign = True

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
        else:
            if is_assign and var and var in self.models:
                self.message( "Resyncing model \"{}\"".format( var ) )
                self.resync_model( var )


    ####################################
    # All do_* command functions go here
    ####################################
    def do_quit( self, args ):
        """Exits the shell"""
        self.message( "Exiting shell" )
        plt.close()
        raise SystemExit


    def do_summary( self, args ):
        """Prints pytorch model summary"""
        try:
            for layer in model.layers:
                self.message( "{}  {}".format( layer._get_name(), layer.size() ) )
        except:
            self.error( sys.exc_info()[ 1 ] )


    def do_set_model( self, args ):
        model_name = args if args else "model"
        model = self.load_from_context( model_name )
        if model is None:
            self.error( "Could not find a model by name \"{}\"".format( model_name ) )
            return

        if not isinstance( model, nn.Module ):
            self.error( "{} is not a valid model" )
            return

        if model_name in self.models:
            self.cur_model = self.models[ model_name ]
        else:
            self.cur_model = ModelMeta( model )
            self.models[ model_name ] = self.cur_model
        self.message( "Context now is-> {}".format( model_name ) )
    

    def do_resync( self, args ):
        if not args:
            self.error( "Please provide a model name" )
            return

        if args not in self.models:
            self.error( "Model \"{}\" not in context".format( args ) )
            return

        self.resync_model( args )


    def do_load_image( self, args ):
        """load a single image"""
        global image
        
        image_path = os.path.join( self.config.image_path, args )
        if not os.path.isfile( image_path ):
            self.error( "Image not found")
            return
        self.message( "Loading image {}".format( image_path ) )
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
            new_state_dict = OrderedDict( [ ( k[ 7: ], v ) for k, v in state_dict.items() 
                                                                        if k.startswith( "module" ) ] )
            model.load_state_dict( new_state_dict )

    do_load_chkp = do_load_checkpoint


    def do_show_image( self, args ):
        img = self.load_from_context( args, default="image" )
        if img is None:
            self.error( "Could not find image" )
            return

        if isinstance( img, torch.Tensor ):
            if img.size( 0 ) == 1:
                img = img.squeeze( 0 )
            if img.size( 2 ) != 3:
                img = img.permute( 1, 2, 0 )
        elif isinstance( img, np.ndarray ):
            if img.shape[ 0 ] == 1:
                img = img[ 0 ]
            if img.shape[ 2 ] != 3:
                img = img.transpose( 1, 2, 0 )
        else:
            self.error( "Unsupported image type" )
            return

        self.fig.imshow( img )
    
    do_show_img = do_show_image


    def do_set_post_process( self, args ):
        if args == "relu":
            fn = torch.nn.ReLU()
        elif args == "mean":
            fn = self.mean4d
        elif args == "max":
            fn = self.max4d
        elif args == "none" or args == "None":
            fn = None
        else:
            fn = self.load_from_context( args )
            if not fn:
                self.error( "Could not find function \"{}\"".format( args ) )
                return

        if not fn:
            self.message( "Removing post processing function" )
            self.data_post_process_fn = None
            return

        if fn and not callable( fn ):
            self.error( "Not a valid function" )
            return

        self.data_post_process_fn = fn
        if not hasattr( self.data_post_process_fn, "__name__" ):
            self.data_post_process_fn.__name__ = args
        self.message( "Post process function is {}".format( self.data_post_process_fn.__name__ ) )

    do_set_postp = do_set_post_process


    def do_show_first_layer_weights( self, args ):
        if args and args not in self.models:
            self.error( "Could not find \"{}\" in context. Please set this model in context first.".format( args ) )
            return
        
        if not args and not self.cur_model:
            self.error( "No default model is set. Please set a model in context first." )
            return

        model_info = self.models[ args ] if args else self.cur_model
        net = model_info.model

        conv = self.find_first_instance( net, layer=nn.Conv2d )
        if not conv:
            self.error( "No Conv2d layer found" )
            return

        w = conv.weight.detach()
        w = w.permute( 0, 2, 3, 1 )

        # Normalize w
        #w = ( w - w.mean() ) / w.std()

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
        self.fig.imshow( grid_w )

    do_show_flw = do_show_first_layer_weights


    def do_show_layer( self, args ):
        img = self.load_from_context( "image" )
        if img is None:
            self.error( "Please load an input image first" )
            return
        
        if args and args not in self.models:
            self.error( "Could not find model {}".format( args ) )
            return
        
        if not args and not self.cur_model:
            self.error( "No default model is set. Please set a model first" )
            return

        model_info = self.models[ args ] if args else self.cur_model
        layer_info = model_info.get_cur_layer_info()

        id, layer = model_info.get_cur_id_layer()
        self.message( "Current layer is {}: {}".format( id, layer ) )
        layer_info.register_forward_hook()
        self.message( "Registered forward hook" )
        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        net = model_info.model
        _ = net( image )

        self.display_layer_data( layer_info, pp_fn=self.data_post_process_fn )


    def do_up( self, args ):
        if not self.cur_model:
            self.error( "Please load a model first" )
            return
        
        if not self.cur_model.up():
            self.message( "Already at top" )
        id, layer = self.cur_model.get_cur_id_layer()
        self.message( "Current layer is {}: {}".format( id, layer ) )


    def do_down( self, args ):
        if not self.cur_model:
            self.error( "Please load a model first" )
            return
        if not self.cur_model.down():
            self.message( "Already at bottom" )
        id, layer = self.cur_model.get_cur_id_layer()
        self.message( "Current layer is {}: {}".format( id, layer ) )


    ###########################
    # Utility functions go here
    ###########################
    def find_first_instance( self, net, layer=nn.Conv2d ):
        for l in net.children():
            if isinstance( l, nn.Conv2d ):
                return l
            ret = self.find_first_instance( l, layer=layer )
            if isinstance( ret, layer ):
                return ret
        return None


    def resync_model( self, name ):
        set_cur_model = False

        cur_model_in_context = self.models[ name ]

        if self.cur_model is cur_model_in_context:
            self.cur_model = None
            set_cur_model = True

        del self.models[ name ]
        
        new_model = self.load_from_context( name )
        if new_model is not None and isinstance( new_model, nn.Module ):
            new_model_info = ModelMeta( new_model )
            self.models[ name ] = new_model_info
            if set_cur_model:
                self.cur_model = new_model_info


    def load_from_context( self, arg, default=None ):
        """This method first processes arg:
            1. If arg is in the global context, return it's value
            2. If arg is not in the global context:
                1. And no default is provided, return None
                2. And a default is provided, process default
        Process default:
            1. Check if default is a string:
                1. If it is, look for it in global context and return it's value
                2. If not in global context, return None
            2. If default is not a string, return default as it is
        """
        if not arg and not default:
            return None

        if arg in self.cur_frame.f_globals:
            return self.cur_frame.f_globals[ arg ]
        elif default is None:
            return None
        elif isinstance( default, str ):
            if default in self.cur_frame.f_globals:
                return self.cur_frame.f_globals[ default ]
            else:
                return None
        else:
            # We are here if args not in context and a non-string default is provided
            return default


    def top_n( self, n, ar ):
        index = np.argpartition( ar, -n )[ -n: ]
        return [ ( i, ar[ i ] ) for i in index ]


    ## FIXBUG: the following function only works when when first dim is 1
    ## May need to be fixed later to deal with batch inputs
    def op_4d( self, data, op ):
        """Only works with 4D tensors for now. Takes the op of last 2 dimensions
        Returns a 2d tensor along the first two dimensions of the input tensor"""
        mean_map = map( lambda x: op( x ).float().item(), data[0][:] )
        return torch.tensor( list( mean_map ) )

    def mean4d( self, data ):
        return self.op_4d( data, op=torch.mean )

    def max4d( self, data ):
        return self.op_4d( data, op=torch.max )


    def display_layer_data( self, layer_info, pp_fn ):
        if layer_info.size( 0 ) != 1:
            self.error( "Unsupported data dimensions" )
            return

        data = layer_info.data()
        data = pp_fn( data ) if pp_fn else data
        
        data = data.squeeze( 0 )
        index = np.arange( data.size( 0 ) )
        # The following statement is invariant to data of dimension ( 1 ).
        # such as a list of tensors. Along the first dimension, replace 
        # the elements with any remaining dimensions with their mean.
        y_data = list( map( lambda x: torch.mean( x ).float().item(), data[ : ] ) )
        
        top5 = self.top_n( 5, y_data )

        ax = self.fig.current_axes()
        ax.bar( index, y_data, align="center", width=1 )
        for i, v in top5:
            ax.text( i, v, "{}".format( i ) )
        ax.set_title( "Histogram of layer {}".format( layer_info.id ) )
        ax.grid()
        self.fig.show_graph()


    ####################################################
    # Helper functions to debugger functionality go here
    ####################################################
    def error( self, err_msg ):
        self.stdout.write( "***{}\n".format( err_msg ) )

    def message( self, msg="", end="\n" ):
        self.stdout.write( msg + end )


    def exec_rc( self ):
        if not self.rc_lines:
            return

        self.message( "\nExecuting rc file" )
        num = 1
        while self.rc_lines:
            line = self.rc_lines.pop( 0 ).strip()
            self.message( "{}: {}".format( num, line ), end="" )
            num += 1
            if not line or "#" in line[ 0 ]:
                self.message()
                continue
            self.onecmd( line )
            self.message( " ...Done" )
        self.message()


    def init_history( self, histfile ):
        try:
            readline.read_history_file( histfile )
        except FileNotFoundError:
            pass        
        readline.set_history_length( 2000 )
        readline.set_auto_history( True )


    def save_history( self, histfile ):
        self.message( "Saving history" )
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
            except ( AttributeError, TypeError ):
                self.error( "----------Error----------" )
                traceback.print_exc()
            except RuntimeError as e:
                self.error( e )


if __name__ == "__main__":
    shell = Shell( config )
    shell.prompt = '>> '
    shell._cmdloop( "Welcome to the shell" )