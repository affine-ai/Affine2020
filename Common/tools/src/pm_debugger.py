#! /usr/bin/env python3

import os, sys, code, traceback
import cmd, readline
import atexit
from collections import OrderedDict
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.models import *
from PIL import Image
from pm_helper_classes import * 


model = None
image = None
out = None

class Config( object ):
    pass 

config = Config()


class Shell( cmd.Cmd ):
    def __init__( self, config ):
        super().__init__()
        self.config = config
        self.dataset = None
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

        if config.dataset:
            self.dataset = Dataset( config.dataset )

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
        """Exits the shell
        """
        self.message( "Exiting shell" )
        plt.close()
        raise SystemExit


    def do_summary( self, args ):
        """Prints pytorch model summary
        """
        try:
            for layer in model.layers:
                self.message( "{}  {}".format( layer._get_name(), layer.size() ) )
        except:
            self.error( sys.exc_info()[ 1 ] )


    def do_nparams( self, args ):
        """Print the total number of parameters in a model
        """
        model_info, _  = self.get_info_from_context( args )
        if model_info is None:
            self.message( "Model \"{}\" not found. Please set the model in context first".format( args ) )
            return

        model = model_info.model
        n =  sum( reduce( lambda x, y: x * y, p.size() ) for p in model.parameters())
        print( "{:,}".format( n ) )

    do_nparam = do_nparams


    def do_load_image( self, args ):
        """Load a single image from the path specified
        Usage: load image [ path ]
        """
        global image
        
        image_path = os.path.join( self.config.image_path, args )
        if not os.path.isfile( image_path ):
            self.error( "Image not found")
            return
        self.message( "Loading image {}".format( image_path ) )
        image = Image.open( image_path )
        transform = transforms.Compose( [ transforms.Resize( ( self.image_size, self.image_size ) ),
                                          transforms.ToTensor() ] )
        image = transform( image ).float().unsqueeze( 0 )


    def do_image_next( self, args ):
        """Load the next available image from a dataset:
        Usage: image next
        
        This command operates on a dataset. A dataset must be configued for this 
        command. If there are no more images available to be loaded, it keeps the 
        last available image.
        The "image" global variable points to the loaded image.
        """ 
        global image
        if self.dataset is None:
            self.message( "Please configure a dataset first" )
            return

        self.dataset.next()
        image = self.dataset.load()
        self.fig.imshow( image )


    def do_load_checkpoint( self, args ):
        """Load a checkpoint file into the model:
        Usage: load checkpoint [ filename ]

        If no file is specified, checkpoint_name specified in the
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
        """Display an image array:
        Usage: show image [ image_var ]
        
        If an (optional) image_var is specified, it's considered as a global 
        variable referencing an image array and tries to show this image.
        If no args are specified, show the image referenced by the
        global "image" variable.
        """ 
        img = self.load_from_global( args, default="image" )
        if img is None:
            self.error( "Could not find image" )
            return

        if not self.fig.imshow( img ):
            self.error( "Unsupported image type" )
            return

    do_show_img = do_show_image


    def do_infer_image( self, args ):
        """Run inference on image:
        Usage: infer image [ model_name ]

        If an optional "model_name" is provided, inference is run
        on that model. The "model_name" must be present in context.
        If no "model_name" is provided, inference is run on the current 
        model set in the context.

        Input image is taken from the global "image" variable.
        """
        model_info, _ = self.get_info_from_context( args )
        if model_info is None:
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "Please load an input image first" )
            return

        net = model_info.model
        out = net( image )
        probs, idxs = F.softmax( out, dim=1 ).topk( 5, dim=1 )
        for idx, prob in zip( idxs[ 0 ], probs[ 0 ] ):
            self.message( "{:<10}{:4.1f}".format( idx.data, prob.data * 100 ) )

    do_infer = do_infer_image


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

        nf, _, _, nc = w.size()
        if nc is 1:
            # This is a grayscale image filter
            w = w.squeeze( 3 )

        s = int( np.floor( np.sqrt( nf ) ) )
        
        # if the number of filters is not a perfect square, we pad 
        # the tensor so that we can display it in a square grid
        if pow( s, 2 ) < nf:
            s += 1
            npad = pow( s, 2, nf )
            w = torch.cat( ( w, torch.ones( ( npad, *w.size()[ 1: ] ) ) ), dim=0 )

        grid_w = torch.cat( tuple( torch.cat( tuple( w[ k ] for k in range( j * s, j * s + s ) ), dim=1 ) 
                                                               for j in range( s ) ), dim=0 )
        self.fig.imshow( grid_w )

    do_show_flw = do_show_first_layer_weights


    def do_show_activations( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        if model_info is None:
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "Please load an input image first" )
            return

        id, layer = layer_info.id, layer_info.layer
        self.message( "Current layer is {}: {}".format( id, layer ) )

        layer_info.register_forward_hook()
        self.message( "Registered forward hook" )

        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        net = model_info.model
        out = net( image )
        self.message( "Out: {}".format( out.argmax() ) )
        title = "{} activations".format( layer_info.id )
        self.display_layer_data( layer_info.data(), title, reduce_fn=self.data_post_process_fn )

    do_show_act = do_show_activations


    def do_show_heatmap( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        
        if model_info is None:
            self.message( "Please set a model in context first" )
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "No input image available" )
            return

        id, layer = model_info.find_last_instance( layer=nn.Conv2d )
        layer_info = model_info.get_layer_info( id, layer )
        layer_info.register_forward_hook()
        self.message( "Registered forward hook" )

        net = model_info.model
        out = net( image )

        idx = out.argmax()
        
        _, fc = model_info.find_last_instance( layer=nn.Linear )
        fc_weights = fc.weight[ idx ].data.numpy()

        activations = layer_info.data()[ 0 ].data.numpy()        
        nc, h, w = activations.shape
        
        cam = fc_weights.reshape( 1, nc ).dot( activations.reshape( nc, h * w ) )
        cam = cam.reshape( h, w )
        cam = ( cam - np.min( cam ) ) / np.max( cam )
        cam = Image.fromarray( cam )

        _, _, h, w = img.size()     
        cam = cam.resize( ( h, w ), Image.BICUBIC )
        cam = transforms.ToTensor()( cam )[ 0 ]
        
        msg = "Model guess: {}".format( idx )
        self.message( msg )
        self.fig.imshow( img, title=msg, dontshow=True )
        self.fig.imshow( cam, persist=True, cmap=cm.jet, norm=colors.Normalize(), alpha=0.5 )

    do_show_heat = do_show_heatmap


    def do_heatmap_next( self, args ):
        global image
        
        if self.dataset is None:
            self.error( "No dataset configured" )
            return

        self.dataset.next()
        image = self.dataset.load()
        self.do_show_heatmap( args=args )
    
    do_heat_next = do_heatmap_next


    def do_show_weights( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        if model_info is None:
            return

        id, layer = layer_info.id, layer_info.layer
        self.message( "Current layer is {}: {}".format( id, layer ) )

        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        try:
            data = layer_info.layer.weight.unsqueeze( 0 )
        except:
            self.error( "Current layer has no weights")
        else:
            title = "{} weights".format( layer_info.id )
            self.display_layer_data( data, title, reduce_fn=self.data_post_process_fn )

    do_show_weight = do_show_weights
    do_show_wei = do_show_weights


    def do_show_grads( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        if model_info is None:
            return
    
        id, layer = layer_info.id, layer_info.layer
        self.message( "Current layer is {}: {}".format( id, layer ) )

        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        try:
            data = layer_info.layer.weight.grad.unsqueeze( 0 )
        except:
            self.error( "Current layer has no gradients" )
        else:
            title = "{} gradients".format( layer_info.id )
            self.display_layer_data( data, title, reduce_fn=self.data_post_process_fn )


    def do_set_model( self, args ):
        model_name = args if args else "model"
        model = self.load_from_global( model_name )
        if model is None:
            self.error( "Could not find a model by name \"{}\"".format( model_name ) )
            return

        if not isinstance( model, nn.Module ):
            self.error( "{} is not a valid model" )
            return

        # If model is already in context, we only need to switch the pointer
        # Otherwise we need to set up the model in the context first
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


    def do_set_post_process( self, args ):
        if args == "relu":
            fn = torch.nn.ReLU()
        elif args == "mean":
            fn = torch.mean
        elif args == "max":
            fn = torch.max
        elif args == "none" or args == "None":
            fn = None
        else:
            fn = self.load_from_global( args )
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


    def do_set_class( self, args ):
        if self.dataset is None:
            self.error( "No dataset is configured" )
        idx = int( args )
        if not self.dataset.set_class( idx ):
            self.error( "Could not set class to {}".format( args ) )


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
        
        new_model = self.load_from_global( name )
        if new_model is not None and isinstance( new_model, nn.Module ):
            new_model_info = ModelMeta( new_model )
            self.models[ name ] = new_model_info
            if set_cur_model:
                self.cur_model = new_model_info


    def load_from_global( self, arg, default=None ):
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


    def display_layer_data( self, data, title, reduce_fn=None ):
        if data.size( 0 ) != 1:
            self.error( "Unsupported data dimensions" )
            return

        reduce_fn = torch.mean if reduce_fn is None else reduce_fn
        
        data = data.squeeze( 0 )
        index = np.arange( data.size( 0 ) )
        # The following statement is invariant to data of dimension ( 1 ).
        # such as a list of tensors. Along the first dimension, replace 
        # the elements with any remaining dimensions with their mean.
        y_data = list( map( lambda x: reduce_fn( x ).float().item(), data[ : ] ) )
        
        top5 = self.top_n( 5, y_data )

        ax = self.fig.current_axes()
        ax.bar( index, y_data, align="center", width=1 )
        for i, v in top5:
            ax.text( i, v, "{}".format( i ) )
        ax.set_title( "Histogram of layer {}".format( title ) )
        ax.grid()
        self.fig.show_graph()


    def get_info_from_context( self, args ):
        if args and args not in self.models:
            self.error( "Could not find model {}".format( args ) )
            return None, None
        
        if not args and not self.cur_model:
            self.error( "No default model is set. Please set a model first" )
            return None, None

        model_info = self.models[ args ] if args else self.cur_model
        layer_info = model_info.cur_layer

        return model_info, layer_info

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