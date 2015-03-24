'''
@author: Yuhuang
@contact: duguyue100@gmail.com

@note: implementation of Long-Short Term Memory
'''

import numpy as np;
import theano;
import theano.tensor as T;
from theano.tensor.shared_randomstreams import RandomStreams;

import telauges.nnfuns as nnf;
import telauges.utils as util;
from telauges.rnn import RNNLayer;
from telauges.layer import Layer;

class LSTM(RNNLayer):
  
  def get_weights(self):
    self.in_gate=Layer(num_in=self.num_in+self.num_hidden,
                       num_hidden=self.num_hidden,
                       activate_mode="sigmoid",
                       clip_gradients=self.clip_gradients);
    
    self.forget_gate=Layer(num_in=self.num_in+self.num_hidden,
                           num_hidden=self.num_hidden,
                           activate_mode="sigmoid",
                           clip_gradients=self.clip_gradients);
    
    self.in_gate_2=Layer(num_in=self.num_in+self.num_hidden,
                         num_hidden=self.num_hidden,
                         activate_mode=self.activate_mode,
                         clip_gradients=self.clip_gradients);
                        
    self.out_gate=Layer(num_in=self.num_in+self.num_hidden,
                        num_hidden=self.num_hidden,
                        activate_mode="sigmoid",
                        clip_gradients=self.clip_gradients);
                        
    self.intern_layers=[self.in_gate, self.forget_gate, self.in_gate_2, self.out_gate];
    
    self.h=util.get_shared_matrix("h", self.num_hidden*2);
    
  def get_activation(self,
                     data,
                     pre_h):
    
    if pre_h.ndim>1:
      prev_c=pre_h[:self.num_hidden, :];
      prev_h=pre_h[self.num_hidden:, :];
    else:
      prev_c=pre_h[:self.num_hidden];
      prev_h=pre_h[self.num_hidden:];
      
    obs=T.concatenate((data, prev_h));
    
    in_gate=self.in_gate.get_output(obs);
    forget_gate=self.forget_gate.get_output(obs);
    in_gate_2=self.in_gate_2.get_output(obs);
    
    next_c=forget_gate*prev_c+in_gate_2*in_gate;
    out_gate=self.out_gate.get_output(obs);
    
    next_h=out_gate*T.tanh(next_c);
    
    return T.concatenate((next_c, next_h));
    
  def get_output(self, 
                 data, 
                 pre_h):
    
    return self.get_activation(data, pre_h);
  
  def postproc_activation(self,
                          x,
                          *args):
    if x.ndim>1:
      return x[self.num_hidden:, :];
    else:
      return x[self.num_hidden:];
    
  @property
  def params(self):
    return ([self.h]+
            [param for layer in self.intern_layers for param in layer.params]);
            
  @params.setter
  def params(self, param_list):
    self.h.set_value(param_list[0].get_value());
    
    start=1;
    for layer in self.intern_layers:
      end=start+len(layer.params);
      layer.params=param_list[start:end];
      start=end;
