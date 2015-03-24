'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: An abstract layer for neural networks. In future, all neural networks layer should extend this class.
The implementation and spirit is taken from https://github.com/JonathanRaiman/theano_lstm
'''

import numpy as np;
import theano;
import theano.tensor as T;
from theano.gof import OpRemove;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;
import telauges.utils as util;

class Layer(object):
  """
  This class is an abstract layer for neural networks.
  """
  
  def __init__(self,
               num_in,
               num_hidden,
               activate_mode="tanh",
               clip_gradients=False):
    """
    @param num_in: dimension of input data (int)
    @param num_hidden: dimension of hidden unit (int)
    @param activate_mode: 5 non-linearity function: tanh, sigmoid, relu, softplus and softmax (string)
    @param clip_gradients: if use clip gradients to control weights (bool)
    """
    
    self.num_in=num_in;
    self.num_hidden=num_hidden;
    self.activate_mode=activate_mode;
    
    # configure activation
    if (self.activate_mode=="tanh"):
      self.activation=nnf.tanh;
    elif (self.activate_mode=="relu"):
      self.activation=nnf.relu;
    elif (self.activate_mode=="sigmoid"):
      self.activation=nnf.sigmoid;
    elif (self.activate_mode=="softplus"):
      self.activation=nnf.softplus;
    elif (self.activate_mode=="softmax"):
      self.activation=nnf.softmax;
    else:
      raise ValueError("Value %s is not a valid choice of activation function"
                       % self.activate_mode);
                       
    self.clip_gradients=clip_gradients;
    self.is_recursive=False;
    
    # init weights
    self.get_weights();
    
  def get_weights(self):
    """
    Get network weights and bias
    
    @return: layer weights and bias
    """
    
    self.W=util.get_shared_matrix("W", self.num_hidden, self.num_in);
    self.b=util.get_shared_matrix("b", self.num_hidden);
    
  def get_pre_activation(self,
                         data):
    """
    Get pre-activation of the data
    
    @param data: assume data is row-wise
    
    @return: a pre-activation matrix
    """
    
    if self.clip_gradients is not False:
      data=self.clip_gradient(data, self.clip_gradients);
    
    return T.dot(data, self.W)+self.b;
  
  def get_activation(self,
                     pre_activation):
    """
    Get activation from pre-activation
    
    @param pre_activation: pre-activation matrix
    
    @return: layer activation
    """
    
    return self.activation(pre_activation);
    
  def get_output(self,
                 data):
    """
    Get layer activation from input data
    
    @param data: input data, assume row-wise
    
    @return layer activation
    """
    
    return self.get_activation(self.get_pre_activation(data));
  
  def clip_gradient(self,
                    data,
                    bound):
    grad_clip=util.GradClip(-bound, bound);
    
    try:
      T.opt.register_canonicalize(OpRemove(grad_clip),
                                  name="grad_clip_%.1f" % (bound));
    except ValueError:
      pass
    
    return grad_clip(data);
  
  @property
  def params(self):
    return (self.W, self.b);
  
  @params.setter
  def params(self, param_list):
    self.W.set_value(param_list[0].get_value());
    self.b.set_value(param_list[1].get_value());
