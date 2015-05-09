'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: A implementation of Convolutional K-means
'''

import numpy as np;

import theano;
import theano.tensor as T;

from telauges.conv_net_layer import ConvNetLayer

class CKMLayer(ConvNetLayer):
  """
  A implementation of Convolutional K-means Layer
  """
  def __init__(self,
               rng,
               feature_maps,
               feature_shape,
               filter_shape,
               pool=False,
               pool_size=(2,2),
               stride_size=None,
               border_mode="valid",
               activate_mode="tanh"):
    """
    Initialize a CKM Layer.
    """
    
    super(CKMLayer, self).__init__(rng=rng,
                                   feature_maps=feature_maps,
                                   feature_shape=feature_shape,
                                   filter_shape=filter_shape,
                                   pool=pool,
                                   pool_size=pool_size,
                                   stride_size=stride_size,
                                   border_mode=border_mode,
                                   activate_mode=activate_mode);
    
#     filter_shape_temp=np.asarray(self.filter_shape);                              
#     weights=np.asarray(rng.uniform(low=0,
#                                    high=1,
#                                    size=(filter_shape_temp[2]*filter_shape_temp[3],
#                                          filter_shape_temp[0])),
#                       dtype='float32');
#     _, v=np.linalg.eig(weights.dot(weights.T)/filter_shape_temp[0]);
#     weights=v.dot(weights);
#     weights=np.reshape(weights, self.filter_shape);
#                                     
#     self.filters = theano.shared(weights,
#                                  borrow=True);
     
    self.filters = theano.shared(np.asarray(rng.uniform(low=-1,
                                                        high=1,
                                                        size=filter_shape),
                                            dtype='float32'),
                                 borrow=True);
                                                               
  def ckm_updates(self):
    """
    This function computes updates of filters and total changes 
    """
    
    feature_shape_temp=np.asarray(self.feature_shape);
    filter_shape_temp=np.asarray(self.filter_shape);
    
    ams_shape=(filter_shape_temp[1],
               feature_shape_temp[0],
               feature_shape_temp[2]-filter_shape_temp[2]+1,
               feature_shape_temp[3]-filter_shape_temp[3]+1);
    
    fms, _=self.get_conv_pool(feature_maps=self.in_feature_maps,
                              feature_shape=self.feature_shape,
                              filters=self.filters,
                              filter_shape=self.filter_shape, 
                              bias=self.b);
    fms=fms.dimshuffle(1,0,2,3);
    #fms=T.nnet.sigmoid(fms);
    
    
    #activation_maps=T.cast(T.ge(fms, T.mean(fms, axis=(2,3), keepdims=True)), dtype="float32");
    argmaxmap=T.argmax(fms, axis=0);
    activation_maps=T.cast(T.ge(fms, 0.5*T.max(fms, axis=(0), keepdims=True)), dtype="float32");
    ams_sum=T.cast(T.sum(activation_maps, axis=(1,2,3), keepdims=True), dtype="float32");
    
    fsn=(feature_shape_temp[1], feature_shape_temp[0], feature_shape_temp[2], feature_shape_temp[3])
    
    update_out, _=self.get_conv_pool(feature_maps=self.in_feature_maps.dimshuffle(1,0,2,3),
                                     feature_shape=fsn,
                                     filters=activation_maps,
                                     filter_shape=ams_shape, 
                                     bias=self.b);
    
    update_out=update_out.dimshuffle(1,0,2,3);
    
    update_out/=(ams_sum+1);
    #T.true_div(update_out, ams_sum);
    
    updates=[(self.filters, 0*self.filters+update_out)];
    
    return updates, T.sum(self.filters), update_out;