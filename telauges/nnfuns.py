'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Mostly neural network activation functions
'''

import theano.tensor as T

def tanh(x):
  """
  Hyperbolic tangent nonlinearity
  """
  return T.tanh(x);

def sigmoid(x):
  """
  Standard sigmoid nonlinearity
  """
  return T.nnet.sigmoid(x);

def softplus(x):
  """
  Softplus nonlinearity
  """
  return T.nnet.softplus(x);

def relu(x):
  """
  Rectified linear unit
  """
  return x*(x>1e-13);

def softmax(x):
  """
  Softmax function
  """
  return T.nnet.softmax(x);
