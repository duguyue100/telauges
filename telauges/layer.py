'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: An abstract layer for neural networks. In future, all neural networks layer should extend this class.
The implementation and spirit is taken from https://github.com/JonathanRaiman/theano_lstm
'''

import theano;
import theano.tensor as T;

import telauges.nnfuns as nnf;
import telauges.utils as util;

class Layer(object):
  """
  This class is an abstract layer for neural networks.
  """
  
  def __init__(self,
               num_in,
               num_hidden,
               is_recursive=False,
               activate_mode="tanh",
               weight_type="none",
               clip_gradients=False,
               clip_bound=1):
    """
    @param num_in: dimension of input data (int)
    @param num_hidden: dimension of hidden unit (int)
    @param is_recursive: True - RNN, False - Feedforward network (bool) 
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
    self.clip_bound=clip_bound;
    self.is_recursive=is_recursive;
    
    # init weights
    self.weight_type=weight_type;
    self.get_weights(self.weight_type);
    
  def get_weights(self,
                  weight_type="none"):
    """
    Get network weights and bias
    
    @param weight_type: "none", "sigmoid", "tanh"
    
    @return: layer weights and bias
    """
    
    self.W=util.get_shared_matrix("W", self.num_hidden, self.num_in, weight_type=weight_type);
    self.b=util.get_shared_matrix("b", self.num_hidden, weight_type=weight_type);
    
  def get_pre_activation(self,
                         X):
    """
    Get pre-activation of the data
    
    @param data: assume data is row-wise
    
    @return: a pre-activation matrix
    """
    
    if self.clip_gradients is True:
      X=theano.gradient.grad_clip(X, -self.clip_bound, self.clip_bound);
    
    return T.dot(X, self.W)+self.b;
  
  def get_activation(self,
                     pre_activation):
    """
    Get activation from pre-activation
    
    @param pre_activation: pre-activation matrix
    
    @return: layer activation
    """
    
    return self.activation(pre_activation);
    
  def get_output(self,
                 X):
    """
    Get layer activation from input data
    
    @param X: input data, assume row-wise
    
    @return layer activation
    """
    
    return self.get_activation(self.get_pre_activation(X));
  
  @property
  def params(self):
    return (self.W, self.b);
  
  @params.setter
  def params(self, param_list):
    self.W.set_value(param_list[0].get_value());
    self.b.set_value(param_list[1].get_value());
