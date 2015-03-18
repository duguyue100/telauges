'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Theano implementation of a MLP Hidden Layer
'''

import numpy as np;
import theano;
import theano.tensor as T;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;

class HiddenLayer(object):
  """
  Theano implementation of MLP Hidden Layer
  """
  
  def __init__(self,
               rng,
               data_in,
               n_in,
               n_out,
               W=None,
               b=None,
               activate_mode="tanh"):
    
    """
    Typical hidden layer of a MLP: units are fully-connected and have
    sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).
    
    @param rng: random number generator for initialize weights (numpy.random.RandomState)
    @param data_in: symbolic tensor of shape (n_examples, n_in) (theano.tensor.dmatrix)
    @param n_in: dimension of input data (int)
    @param n_out: dimension of hidden unit (int)
    @param activate_mode: 5 non-linearity function: tanh, sigmoid, relu, softplus and softmax (string)    
    """
    
    self.input=data_in;
    self.n_in=n_in;
    self.n_out=n_out;
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
    else:
      raise ValueError("Value %s is not a valid choice of activation function"
                       % self.activate_mode);

    if W is None:
      W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                        high=np.sqrt(6. / (n_in + n_out)),
                                        size=(n_in, n_out)),
                            dtype='float32');
      if (self.activate_mode=="sigmoid"):
        W_values *= 4;

      W = theano.shared(value=W_values, name='W', borrow=True);

    if b is None:
      b_values = np.zeros((n_out,), dtype='float32');
      b = theano.shared(value=b_values, name='b', borrow=True);

    self.W = W;
    self.b = b;

    self.weighted_sum=self.get_pre_activation(self.input);
    self.output=self.get_activation(self.weighted_sum);

    self.params = [self.W, self.b];
    
  def get_weighted_sum(self,
                       data_in,
                       W=None,
                       b=None):
    """
    Get weighted sum of the input
    """
    return T.dot(data_in, W)+b;
  
  def get_pre_activation(self,
                   data_in):
    """
    Get weighted sum using self weight and bias
    """
    
    return self.get_weighted_sum(data_in, W=self.W, b=self.b);
  
  def get_activation(self,
                     s):
    """
    Get layer activation based on activation function
    """
    
    return self.activation(s);
  
  def get_output(self, x):
    return self.get_activation(self.get_pre_activation(x));

class SoftmaxLayer(HiddenLayer):
  """
  Softmax Layer implementation
  """
  
  def __init__(self,
               rng,
               data_in,
               n_in,
               n_out,
               W=None,
               b=None):
    
    super(SoftmaxLayer, self).__init__(rng=rng,
                                       data_in=data_in,
                                       n_in=n_in,
                                       n_out=n_out,
                                       W=W,
                                       b=b,
                                       activate_mode="softmax");
                                       
    self.prediction=T.argmax(self.output, axis=1);
                                       
  def cost(self,
           y):
    """
    Cost of softmax regression
    """
    
    return T.nnet.categorical_crossentropy(self.output, y).mean();
  
  def errors(self,
             y):
    """
    Difference between true label and prediction
    """
    
    return T.mean(T.neq(self.prediction, y));
  
class AutoEncoder(object):
  """
  An implementation of naive Auto-encoder
  """
  
  def __init__(self,
               rng,
               data_in,
               n_vis,
               n_hidden,
               encode_activate_mode="tanh",
               decode_activate_mode="tanh"):
    """
    Init a naive Auto-encoder without bounding weights
    """
    
    self.rng=rng;
    self.theano_rng=RandomStreams(rng.randint(2 ** 30));
    self.input=data_in;
    self.n_vis=n_vis;
    self.n_hidden=n_hidden;
    
    self.encode_activate_mode=encode_activate_mode;
    self.decode_activate_mode=decode_activate_mode;
    
    self.encode_layer=HiddenLayer(rng=self.rng,
                                  data_in=self.input,
                                  n_in=self.n_vis,
                                  n_out=self.n_hidden,
                                  activate_mode=self.encode_activate_mode);
    
    self.decode_layer=HiddenLayer(rng=self.rng,
                                  data_in=self.encode_layer.output,
                                  n_in=self.n_hidden,
                                  n_out=self.n_vis,
                                  activate_mode=self.decode_activate_mode);
                                  
    self.parms=self.encode_layer.params;
                                  
  def get_feature(self, x):
    return self.encode_layer.get_output(x);
  
  def get_decode_from_feature(self, x):
    return self.decode_layer.get_output(x);
  
  def get_corruption_input(self,
                           x,
                           corruption_level):
    return self.theano_rng.binomial(size=x.shape, n=1,
                                    p=1 - corruption_level,
                                    dtype="float32") * x;
  
  def get_cost(self, x, y):
    return -T.sum(x * T.log(y) + (1-x)* T.log(1 - y), axis=1);
  
  def get_updates(self,
                  learning_rate,
                  corruption_level=None,
                  L1_rate=0.000,
                  L2_rate=0.000,):
    
    if corruption_level is not None:
      x=self.get_corruption_input(self.input, corruption_level);
      y=self.decode_layer.get_output(self.encode_layer.get_output(x));
    else:
      y=self.decode_layer.output;
      
    #cost=T.sum(T.pow(T.sub(self.decode_layer.output, x),2), axis=1);
    
    cost=self.get_cost(self.input, y);    
    cost=T.mean(cost);
    
    params=self.encode_layer.params+self.decode_layer.params;
    gparams=T.grad(cost, params);

    updates=[(param_i, param_i-0.1*grad_i)
             for param_i, grad_i in zip(params, gparams)];
             
    return cost, updates;