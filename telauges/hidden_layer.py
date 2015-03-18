'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Theano implementation of a MLP Hidden Layer
'''

import numpy as np;
import theano;
import theano.tensor as T;

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
    
    self.input = data_in;
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