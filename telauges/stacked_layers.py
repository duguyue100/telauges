'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: an generic layer for stacking all kinds of layers
'''

import theano;
import theano.tensor as T;

import telauges.optimization as opt;

class StackedLayers(object):
  
  def __init__(self,
               num_in,
               layers=None,
               clip_gradients=False):
    """
    
    @param num_in: input size (int)
    @param layers: list of layer configuration (list)
    @param clip_gradients: if use clip gradients (bool)
    """
    
    if layers is None:
      layers=[];
      
    self.layers=layers;
    self.num_in=num_in;
    self.clip_gradients=clip_gradients;
    
  def fprop(self,
            data,
            prev_hiddens=None,
            dropout=None):
    """
    Forward propagation
    
    @param data: input data
    @param pre_hiddens: 
    @param dropout: list of dropout masks
    """
    
    if dropout is None:
      dropout=[];
      
    if prev_hiddens is None:
      prev_hiddens=[(T.repeat(T.shape_padleft(layer.h),
                              data.shape[0],
                              axis=0)
                     if data.dim>1 else layer.h)
                    if hasattr(layer, "h") else None
                    for layer in self.layers];
      
    out=[];
    layer_input=data;
    
    for k, layer in enumerate(self.layers):
      level_out=layer_input;
      
      if len(dropout)>0:
        level_out=opt.apply_dropout(level_out, dropout[k]);
        
      if layer.is_recursive:
        level_out=layer.get_output(level_out, prev_hiddens[k]);
      else:
        level_out=layer.get_output(level_out);
        
      out.append(level_out);
        
      if hasattr(layer, "postproc_activation"):
        if layer.is_recursive:
          level_out=layer.postproc_activation(level_out, layer_input, prev_hiddens[k]);
        else:
          level_out=layer.postproc_activation(level_out, layer_input);    
      
    return out;