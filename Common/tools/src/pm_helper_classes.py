import os, sys
from pathlib import Path
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from PIL import Image


# Disable the top menubar on plots
matplotlib.rcParams[ "toolbar" ] = "None"


class GraphWindow( object ):
    def __init__( self ):
        self.fig = plt.figure()
        self.window_title = "PM Debug"
        self.cur_ax = None
        self.num_windows = 1
        self.cur_window = None
        self.mode = None
        self.set_window_title()
        self.set_mode( "single" )

    def set_window_title( self, title=None ):
        if title is None:
            t = self.window_title
        else:
            t = "{} ( {} )".format( self.window_title, title )
        self.fig.canvas.set_window_title( t )

    def reset_window( self ):
        for ax in self.fig.axes:
            self.fig.delaxes( ax )

    def set_mode( self, mode ):
        prev_mode = self.num_windows
        if mode == "dual":
            self.num_windows = 2
        else:
            self.num_windows = 1
        if prev_mode is not self.num_windows:
            self.cur_ax = None
            self.cur_window = None

    def current_axes( self, persist=False ):
        """Return the current axis to draw on.
        If persist flag is set, always return the current axis.
        If persist flag is not set:
            In 'single' window mode, clear the window before returning the axes
            In 'dual' window mode, return the axes for the next window
        """
        if self.cur_ax is None:
            self.reset_window()
            self.fig.subplots( 1, self.num_windows )
            self.cur_window = 0
            self.cur_ax = self.fig.axes[ 0 ]
        else:
            if not persist:
                if self.num_windows is 1:
                    self.cur_ax.clear()
                else:
                    self.cur_window = self.cur_window ^ 1
                    self.cur_ax = self.fig.axes[ self.cur_window ]
        return self.cur_ax

    def window( self, num ):
        if not isinstance( num, int ):
            return False
        if num > self.num_windows:
            return False
        return self.fig.axes[ num ]

    def imshow( self, image, persist=False, dontshow=False, title=None, **kwargs ):
        if isinstance( image, np.ndarray ):
            image = torch.Tensor( image )
        
        if isinstance( image, torch.Tensor ):
            if image.dim() == 4 and image.size( 0 ) == 1:
                # extract image
                image = image.squeeze( 0 )
            if image.dim() is 3 and image.size()[ -1 ] is not 3:
                # image in ( C x H x W ) format
                image = image.permute( 1, 2, 0 )
                # remove the extra dimension if it is a grayscale image
                image = image.squeeze( 2 )
        else:
            return False

        if image.dim() is 2 and "cmap" not in kwargs:
            # it is a grayscale image, set the color map correctly
            kwargs[ "cmap" ] = "gray"

        ax = self.current_axes( persist=persist )
        if title is not None:
            ax.set_title( title )
        ax.imshow( image, **kwargs )
        ax.figure.canvas.draw()
        if not dontshow:
            self.fig.show()
        return True

    def show_graph( self, ax=None, aspect="auto" ):
        ax = self.cur_ax if ax is None else ax
        ax.set_aspect( aspect )
        ax.figure.canvas.draw()
        self.fig.show()

    def close( self ):
        plt.close()


class Dataset( object ):
    def __init__( self, path ):
        self.data = None
        self.data_path = Path( path )
        self.cur_class = 0
        self.cur_dir = self.data_path
        self.cur_image_file = None
        self.file_iter = None
        self.image_size = 224
        self.my_transforms = [ transforms.Resize( ( self.image_size, self.image_size ) ) ]
        self.set_class( self.cur_class )

    def reset_class( self ):
        self.file_iter.close()
        self.file_iter = None
        self.cur_image_file = None

    def set_class( self, label ):
        listdir = [ d for d in self.data_path.iterdir() if d.is_dir() ]
        listdir.sort()
        label = int( label )
        try:
            dir = listdir[ label ]
        except:
            return False
        path = self.data_path / dir
        if path.is_dir():
            self.cur_class = label
            self.cur_dir = path
            if self.file_iter:
                self.reset_class()
            return True
        else:
            return False

    def next( self ):
        if self.file_iter is None:
            self.file_iter = iter( Path( self.cur_dir ).iterdir() )

        image_file = next( self.file_iter )
        if image_file.is_file():
            self.cur_image_file = image_file
            return str( image_file ), image_file.name
        else:
            return None, None

    def load( self ):
        image = Image.open( self.cur_image_file )
        transform = transforms.Compose( self.my_transforms )
        return transforms.ToTensor()( transform( image ) ).unsqueeze( 0 )

    def suffix( self, suffix ):
        if suffix in ( "train", "val", "validation" ):
            self.data_path = self.data_path.parent / suffix
            self.set_class( self.cur_class )
            return True
        else:
            return False

    def add_transform( self, t, index=1 ):
        self.my_transforms.append( t )

    def del_transform( self, index ):
        self.my_transforms.pop( index )

class ModelMeta( object ):
    def __init__( self, model, name ):
        self.model = model
        self.name = name
        self.cur_layer = None
        self.layers = OrderedDict()
        self.init_layer()

    def init_layer( self ):
        id, layer = self.find_last_instance( layer=nn.ReLU )
        layer_info = LayerMeta( layer, id )
        self.layers[ tuple( id ) ] = layer_info
        self.cur_layer = layer_info

    def get_cur_id_layer( self ):
        if not self.cur_layer:
            self.init_layer()
        return self.cur_layer.id, self.cur_layer.layer

    def get_layer_info( self, id, layer ):
        if tuple( id ) in self.layers:
            layer_info = self.layers[ tuple( id ) ]
        else:
            layer_info = LayerMeta( layer, id )
            self.layers[ tuple( id ) ] = layer_info
        return layer_info

    def up( self ):
        return self.traverse_updown( dir=-1 )

    def down( self ):
        return self.traverse_updown( dir=1 )

    def traverse_updown( self, dir ):
        id = self.cur_layer.id
        new_id, new_layer = self.find_instance_by_id( id, dir=dir )
        
        if not new_id:
            return False

        id, layer = new_id, new_layer

        self.cur_layer = self.get_layer_info( id, layer )
        return True

    def find_instance_by_id( self, key, dir, net=None ):
        """This function implements a depth first search that terminates 
        as soon as a sufficient condition is met
        """
        net = self.model if net is None else net

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


    def find_last_instance( self, layer=nn.Conv2d, net=None, cur_frame=[], found_frame=[] ):
        """This method does a depth first search and finds the last instance of the
        specifiied layer in the tree
        """
        net = self.model if net is None else net
        
        found = None
        ret_found = None

        for i, l in enumerate( net.children() ):
            cur_frame.append( i )

            if isinstance( l, layer ):
                found = l
                if cur_frame > found_frame:
                    found_frame = cur_frame.copy()

            found_frame, ret_found = self.find_last_instance( layer=layer, net=l,
                                                            cur_frame=cur_frame, 
                                                            found_frame=found_frame )
            if isinstance( ret_found, layer ):
                found = ret_found
            cur_frame.pop()
        return found_frame, found


class LayerMeta( object ):
    def __init__( self, layer, id=[] ):
        self.layer = layer
        self.id = id
        self.out = None
        self.post_process_fn = None
        self.cur_filter = None
        self.num_filters = None
        self.init_num_filters()

    def init_num_filters( self ):
        try:
            self.num_filters = self.layer.weight.size()[ 0 ]
            self.cur_filter = 0
        except:
            self.num_filters = None

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

    def filter_inc( self, n ):
        if self.cur_filter is None:
            return
        self.cur_filter += n
        if self.cur_filter >= self.num_filters:
            self.cur_filter = 0
        if self.cur_filter < 0:
            self.cur_filter = 0

    def close( self ):
        self.fhook.remove()
