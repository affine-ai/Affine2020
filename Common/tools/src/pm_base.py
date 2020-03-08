import os, sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from pm_helper_classes import GraphWindow, ModelMeta


class ShellBase( object ):
    def __init__( self ):
        super().__init__()
        self.stdout = sys.stdout
        self.image_size = 224
        self.device = "cpu"
        self.models = {}
        self.cur_model = None
        self.data_post_process_fn = None
        self.compare = None
        self.verbose_help = True
        self.quiet = False
        self.stack = []
        self.cur_frame = sys._getframe().f_back
        self.fig = GraphWindow()

    def set_model( self, name, model ):
        # If model is already in context, we only need to switch the pointer
        # Otherwise we need to set up the model in the context first
        if name in self.models:
            self.cur_model = self.models[ name ]
        else:
            self.cur_model = ModelMeta( model, name )
            self.models[ name ] = self.cur_model

    def resync_model( self, name ):
        set_as_cur_model = False

        cur_model_in_context = self.models[ name ]

        if self.cur_model is cur_model_in_context:
            self.cur_model = None
            set_as_cur_model = True

        del self.models[ name ]
        
        new_model = self.load_from_global( name )
        if new_model is not None and isinstance( new_model, nn.Module ):
            new_model_info = ModelMeta( new_model, name )
            self.models[ name ] = new_model_info
            if set_as_cur_model:
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

    def in_place_eval( self, args ):
        locals = self.cur_frame.f_locals
        globals = self.cur_frame.f_globals
        try:
            code = compile( args, "<string>", "eval" )
            out = eval( code, globals, locals )
        except:
            self.error( sys.exc_info()[ 0 ] )
            return None
        else:
            return out

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

    def display_bargraph( self, data, title, reduce_fn=None ):
        if data.size( 0 ) != 1:
            self.error( "Unsupported data dimensions" )
            return

        reduce_fn = torch.mean if reduce_fn is None else reduce_fn        
        data = data.squeeze( 0 )
        index = np.arange( data.size( 0 ) )
        # The following statement is invariant to data of dimension ( 1 ).
        # such as a list of tensors. Along the first dimension, replace 
        # the elements with any remaining dimensions with their mean.
        y_data = torch.tensor( list( map( lambda x: reduce_fn( x ).float().item(), data[ : ] ) ) )
        
        val, id = y_data.topk( 5, dim=0, largest=True, sorted=True )

        ax = self.fig.current_axes().ax
        ax.set_title( title )
        ax.bar( index, y_data, align="center", width=1 )
        for i, v in zip( id, val ):
            ax.text( i, v, "{}".format( i ) )
        ax.grid()
        self.fig.show_graph( ax )

    def show_weights_as_grid( self, weight, title=None, cursor=None, zoom=None ):
        if not isinstance( weight, torch.Tensor ):
            return False
        nf, nc, h, w = weight.size()
        # if the number of filters is not a perfect square, we pad 
        # the tensor so that we can display it in a square grid
        s = int( np.floor( np.sqrt( nf ) ) )
        if pow( s, 2 ) < nf:
            s += 1
            npad = pow( s, 2, nf )
            weight = torch.cat( ( weight, torch.ones( ( npad, nc, h, w ) ) ), dim=0 ) #pylint: disable=no-member
        
        grid = torchvision.utils.make_grid( weight, nrow=s, padding=1 )
        if cursor is not None and not zoom:
            x, y = ( h + 1 ) * ( cursor % s ), ( w + 1 ) * ( cursor // s )
            rect = ( ( x, y ), w + 2, h + 2 )
            self.fig.imshow( grid, title=title, rect=rect )
        elif cursor is not None and zoom:
            self.fig.imshow( weight[ cursor ] )
        else:
            self.fig.imshow( grid, title=title )

    def error( self, err_msg ):
        self.stdout.write( "*** {}\n".format( err_msg ) )

    def message( self, msg="", end="\n" ):
        if not self.quiet:
            self.stdout.write( msg + end )

    def help( self, msg="", end="\n" ):
        if self.verbose_help:
            self.stdout.write( "Info: " + msg + end )

    def close( self ):
        self.fig.close()