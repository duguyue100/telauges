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
               h_bias=None):
    
    self.rng=rng;
    self.theano_rng=RandomStreams(self.rng.randint(2**30));
    self.input=data_in;
    self.n_vis=n_vis;
    self.n_hidden=n_hidden;
    
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
    
  