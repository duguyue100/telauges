'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: a simple type recurrent neural networks
'''

import numpy as np;
import theano;
import theano.tensor as T;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;
import telauges.utils as util;
from telauges.layer import Layer;

class RNNLayer(Layer):
  """
  Simple type recurrent neural network layer
  """
  
  def __init__(self, *args, **kwargs):
    super(RNNLayer, self).__init__(*args, **kwargs);
    
    self.is_recursive=True;
    
  def get_weights(self):
    """
    get weights, bias and initial hidden state 
    """
    
    self.W=util.get_shared_matrix("W", self.num_hidden, self.num_in+self.num_hidden);
    self.b=util.get_shared_matrix("b", self.num_hidden);
    self.h=util.get_shared_matrix("h", self.num_hidden);
    
  def get_pre_activation(self,
                         data,
                         pre_h):
    
    if self.clip_gradients is not False:
      data=self.clip_gradient(data, self.clip_gradients);
      pre_h=self.clip_gradient(pre_h, self.clip_gradients);
      
    return T.dot(T.concatenate([data, pre_h]), self.W)+self.b;
  
  def get_output(self, 
                 data,
                 pre_h):
    
    return self.get_activation(self.get_pre_activation(data, pre_h));
  
  @property
  def params(self):
    return [self.W, self.b, self.h];
  
  @params.setter
  def params(self,
             param_list):
    self.W.set_value(param_list[0].get_value());
    self.b.set_value(param_list[1].get_value());
    self.h.set_value(param_list[2].get_value());
