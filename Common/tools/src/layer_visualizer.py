
from pm_base import ShellBase
import torch.nn as nn

class LayerVisualizer( ShellBase ):
    def __init__( self, model_info ):
        super().__init__()
        self.model_info = model_info
        self.fnum = 0
        self.row = 0
        self.col = 0
        self.grid_size = 0
        self._weights = None
        self._data = None
        self.model_info.set_first_layer()

    def set_image( self, image ):
        self.image = image

    def start( self ):
        try:
            self.evaluate()
        except:
            raise

        self.fig.set_mode( "dual" )        
        try:
            self.render( self.fnum )
            self.fig.on_event( "key_press_event", self.keypress )
        except:
            raise
        finally:
            self.fig.set_mode( "single" )   

    def render( self, fnum ):
        self.show_weights_as_grid( self._weights, cursor=fnum )
        self.fig.imshow( self._data[ fnum ] )

    def evaluate( self, ):
        self.fnum = 0
        self.row = 0
        self.col = 0
        self.model_info.cur_layer.register_forward_hook()
        _ = self.model_info.model( self.image )
        self._data = self.model_info.cur_layer.data().squeeze( 0 )
        self._weights = self.model_info.cur_layer.layer.weight.detach().clone()

        # If the layer has more than 3 output filters, reduce them so they
        # can be displayed
        if self._weights.size( 1 ) > 3:
            self._weights = torch.mean( self._weights, dim=1, keepdim=True )

        self.grid_size = self.compute_grid_size( self._weights.size( 0 ) )
        self.message( "{} {}".format( self.model_info.cur_layer.id, self.model_info.cur_layer.layer ) )


    def keypress( self, event ):
        _row = row = self.row
        _col = col = self.col
        if event.key == "left":
            _col = col - 1 if col > 0 else col
        elif event.key == "right":
            _col = col + 1 if col < self.grid_size - 1 else col
        elif event.key == "up":
            _row = row - 1 if row > 0 else row
        elif event.key == "down":
            _row = row + 1 if row < self.grid_size - 1 else row
        elif event.key == "pageup":
            self.cur_model.traverse_updown( dir=-1, type=nn.Conv2d )
            self.evaluate()
        elif event.key == "pagedown":
            self.cur_model.traverse_updown( dir=1, type=nn.Conv2d )
            self.evaluate()
        elif event.key == 'q' or event.key == 'Q':
            self.fig.stop_event_loop()
            return
        fnum = _row * self.grid_size + _col
        if fnum < self._weights.size( 0 ):
            self.row, self.col = _row, _col
            self.render( fnum )
    
     
