'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: An implementation of Restricted Boltzmann Machine (RBM)
'''

import numpy as np;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;

class RBM(object):
  """
  An implementation of Restricted Boltzmann Machine
  """
  
  def __init__(self,
               rng,
               data_in,
               n_vis,
               n_hidden,
               W=None,
               v_bias=None,
               h_bias=None,
               activate_mode="tanh"):
    
    self.rng=rng;
    self.theano_rng=RandomStreams(self.rng.randint(2**30));
    self.input=data_in;
    self.n_vis=n_vis;
    self.n_hidden=n_hidden;
    self.activate_mode=activate_mode;
    
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
    elif (self.activate_mode=="linear"):
      self.activation=nnf.linear;
    else:
      raise ValueError("Value %s is not a valid choice of activation function"
                       % self.activate_mode);
    
    if W is None:
      initial_W = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_vis)),
                                         high=4 * np.sqrt(6. / (n_hidden + n_vis)),
                                         size=(n_vis, n_hidden)),
                             dtype="float32");
                             
      W=theano.shared(value=initial_W, name="W", borrow=True);
      
    if v_bias is None:
      v_bias=theano.shared(value=np.zeros(n_vis, dtype="float32"),
                           name="v_bias",
                           borrow=True);
    if h_bias is None:
      h_bias=theano.shared(value=np.zeros(n_hidden, dtype="float32"),
                           name="h_bias",
                           borrow=True);
                           
    self.W=W;
    self.v_bias=v_bias;
    self.h_bias=h_bias;
    
    self.params=[self.W, self.v_bias, self.h_bias];
    
  def prop_up(self,
              vis):
    
    pre_activation=T.dot(vis, self.W)+self.h_bias;
    
    return [pre_activation, self.activation(pre_activation)];
  
  def prop_down(self,
                hidden):
    
    pre_activation=T.dot(hidden, self.W.T)+self.v_bias;
    
    return [pre_activation, self.activation(pre_activation)];
  
  