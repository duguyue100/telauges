'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: implementation of learning and optimization algorithms
'''

import numpy as np;
from collections import OrderedDict;

import theano;
import theano.tensor as T;

theano_rng=T.shared_randomstreams.RandomStreams(1234);

def get_gd_updates(cost,
                   params,
                   updates=None,
                   max_norm=5.0,
                   learning_rate=0.01,
                   eps=1e-6,
                   rho=0.95,
                   method="adadelta"):
  
  """
  Get gradient descent updates
  
  @param cost: cost expression
  @param params: list of variables
  @param updates: 
  @param max_norm: cap on excess gradient (float)
  @param learning_rate: learning rate (float)
  @param eps: numerical stability value to not divided by zero (float)
  @param rho: adadelta hyperparameter (float)
  @param method: "adagrad", "adadelta", "sgd" 
  """
  
  learning_rate=theano.shared(value=learning_rate,
                              dtype="float32")
  rho=theano.shared(value=rho,
                    dtype="float32");
                    
  if max_norm is not None and max_norm is not False:
    max_norm=theano.shared(value=max_norm,
                           dtype="float32");
                           
  gsums=[theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method=="adadelta" or method=="adagrad") else None for param in params];
  xsums=[theano.shared(np.zeros_like(param.get_value(borrow=True))) if (method=="adadelta") else None for param in params];
  
  gparams=T.grad(cost, params);
  
  if updates is None:
    updates=OrderedDict();
    
  for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
    
    if max_norm is not None and max_norm is not False:
      grad_norm=gparam.norm(L=2);
      gparam=(T.minimum(max_norm, grad_norm)/grad_norm)*gparam;
      
    if method=="adadelta":
      updates[gsum]=T.cast(x=rho*gsum+(1.-rho)*(gparam**2),
                           dtype="float32");
      dparam=-T.sqrt((xsum+eps)/(updates[gsum]+eps))*gparam;
      updates[xsum]=T.cast(x=rho*xsum+(1.-rho)*(dparam**2),
                           dtype="float32");
      updates[param]=T.cast(x=param+dparam,
                            dtype="float32");
    elif method=="adagrad":
      updates[gsum]=T.cast(x=gsum+(gparam**2),
                           dtype="float32");
      updates[param]=T.cast(x=param-learning_rate*(gparam/(T.sqrt(updates[gsum]+eps))),
                            dtype="float32");
    else:
      updates[param]=param-gparam*learning_rate;
      
    if method=="adadelta":
      learning_rate=rho;
      
    return updates, gsums, xsums, learning_rate, max_norm;
  
def dropout(shape, prob):
  """
  Get dropout mask
  """
  
  mask=theano_rng.binominal(n=1, p=1-prob, size=shape);
  
  return T.cast(x=mask, dtype="float32");

def multi_dropout(shapes, dropout=0.):
  
  return [dropout(shape, dropout) for shape in shapes];

def apply_dropout(x, mask):
  if mask is not None:
    return mask*x;
  else:
    return x;