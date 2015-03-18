'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: An implementation of Convolutional Auto-encoder
'''

import numpy as np;
import theano;
import theano.tensor as T;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;
from telauges.conv_net_layer import ConvNetLayer;

class ConvAE(object):
  """
  An implemenation of naive ConvNet Auto-encoder
  """
  
  def __init__(self,
               rng,
               feature_maps,
               feature_shape,
               filter_shape,
               encode_activate_mode="tanh",
               decode_activate_mode="tanh"):
    """
    Init a ConvAE
    
    @param rng: random number generator for initializing weights (numpy.random.RandomState)
    @param feature_maps: symbolic tensor of shape feature_shape (theano.tensor.dtensor4)
    @param feature_shape: (batch size, number of input feature maps,
                           image height, image width) (tuple or list of length 4)
    @param filter_shape: (number of filters, number of input feature maps,
                          filter height, filter width) (tuple or list of length 4)
    @param border_mode: convolution mode
                        "valid" for valid convolution;
                        "full" for full convolution; (string)  
    @param activate_mode: activation mode,
                          "tanh" for tanh function;
                          "relu" for ReLU function;
                          "sigmoid" for Sigmoid function;
                          "softplus" for Softplus function (string)
    
    """
    
    self.rng=rng;
    self.feature_maps=feature_maps;
    self.feature_shape=feature_shape;
    self.filter_shape=filter_shape;
    self.encode_activate_mode=encode_activate_mode;
    self.decode_activate_mode=decode_activate_mode;
    
    self.encode_layer=ConvNetLayer(rng=self.rng,
                                   feature_maps=self.feature_maps,
                                   feature_shape=self.feature_shape,
                                   filter_shape=self.filter_shape,
                                   border_mode="full",
                                   activate_mode=self.encode_activate_mode);
                                   
    feature_shape_temp=np.asarray(feature_shape);
    filter_shape_temp=np.asarray(filter_shape);
    feature_shape_decode=(feature_shape_temp[0],
                          filter_shape_temp[0],
                          feature_shape_temp[2]+filter_shape_temp[2]-1,
                          feature_shape_temp[3]+filter_shape_temp[3]-1,);
                            
    filter_shape_decode=(feature_shape_temp[1],
                         filter_shape_temp[0],
                         filter_shape_temp[2],
                         filter_shape_temp[3]);
                                   
    self.decode_layer=ConvNetLayer(rng=self.rng,
                                   feature_maps=self.encode_layer.out_feature_maps,
                                   feature_shape=feature_shape_decode,
                                   filter_shape=filter_shape_decode,
                                   border_mode="valid",
                                   activate_mode=self.decode_activate_mode);
    
    self.params=self.encode_layer.params;
    
  def get_feature(self, x):
    return self.encode_layer.get_output(x);
  
  def get_decode_from_feature(self, x):
    return self.decode_layer.get_output(x);
  
  def get_cost(self, x, y):
    return -T.sum(x * T.log(y) + (1-x)* T.log(1 - y), axis=1);
  
  def get_updates(self,
                  learning_rate,
                  corruption_level=None,
                  L1_rate=0.000,
                  L2_rate=0.000):
    
    if corruption_level is not None:
      x=self.get_corruption_input(self.input, corruption_level);
      y=self.decode_layer.get_output(self.encode_layer.get_output(x));
    else:
      y=self.decode_layer.out_feature_maps;
      
    #cost=T.sum(T.pow(T.sub(self.decode_layer.output, x),2), axis=1);
    
    cost=self.get_cost(self.feature_maps, y);    
    cost=T.mean(cost);
    
    params=self.encode_layer.params+self.decode_layer.params;
    gparams=T.grad(cost, params);

    updates=[(param_i, param_i-0.1*grad_i)
             for param_i, grad_i in zip(params, gparams)];
             
    return cost, updates;