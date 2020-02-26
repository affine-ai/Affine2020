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
        self.cur_ax = None

    def reset_window( self ):
        for ax in self.fig.axes:
            self.fig.delaxes( ax )

    def current_axes( self, persist=False ):
        if persist and self.cur_ax is not None:
            return self.cur_ax
        else:
            self.reset_window()
            self.cur_ax = self.fig.subplots( 1, 1 )
            return self.cur_ax

    def imshow( self, image, persist=False, dontshow=False, **kwargs ):
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
        ax.imshow( image, **kwargs )
        self.fig.canvas.draw()
        if not dontshow:
            self.fig.show()
        return True

    def show_graph( self ):
        self.fig.canvas.draw()
        self.fig.show()

    def close( self ):
        plt.close()

class Dataset( object ):
    def __init__( self, path ):
        self.data = None
        self.data_path = path
        self.cur_dir = path
        self.cur_image = None
        self.file_iter = None
        self.set_class( 0 )
        self.image_size = 224
        self.transforms = [ transforms.Resize( ( self.image_size, self.image_size ) ),
                            transforms.ToTensor() ]

    def reset_class( self ):
        self.file_iter.close()
        self.file_iter = None
        self.cur_image = None

    def set_class( self, label ):
        listdir = os.listdir( self.data_path )
        listdir.sort()
        label = int( label )
        try:
            dir = listdir[ label ]
        except:
            return False
        path = os.path.join( self.data_path, dir )
        if os.path.isdir( path ):
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
            self.cur_image = image_file
            return str( image_file ), image_file.name
        else:
            return None, None

    def load( self ):
        image = Image.open( self.cur_image )
        transform = transforms.Compose( self.transforms )
        return transform( image ).float().unsqueeze( 0 )

    def suffix( self, suffix ):
        pass

    def add_transform( self, t, index=1 ):
        self.transforms.insert( index, t )

    def del_transform( self, index ):
        self.transforms.pop( index )

class ModelMeta( object ):
    def __init__( self, model ):
        self.model = model
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