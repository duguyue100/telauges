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
  
  def sample_h_given_v(self, v0_sample):
    
    pre_h1, h1_mean=self.prop_up(v0_sample);
    
    h1_sample=self.theano_rng.binomial(size=h1_mean.shape,
                                       n=1,
                                       p=h1_mean,
                                       dtype="float32");
                                       
    return [pre_h1, h1_mean, h1_sample];
  
  def prop_down(self,
                hidden):
    
    pre_activation=T.dot(hidden, self.W.T)+self.v_bias;
    
    return [pre_activation, self.activation(pre_activation)];
  
  def sample_v_given_h(self, h0_sample):
    
    pre_v1, v1_mean=self.prop_down(h0_sample);
    
    v1_sample=self.theano_rng.binomial(size=v1_mean.shape,
                                       n=1,
                                       p=v1_mean,
                                       dtype="float32");
                                       
    return [pre_v1, v1_mean, v1_sample];
  
  def gibbs_hvh(self, h0_sample):
    
    pre_v1, v1_mean, v1_sample=self.sample_v_given_h(h0_sample);
    pre_h1, h1_mean, h1_sample=self.sample_h_given_v(v1_sample);
    
    return [pre_v1, v1_mean, v1_sample,
            pre_h1, h1_mean, h1_sample];
            
  def gibbs_vhv(self, v0_sample):
    
    pre_h1, h1_mean, h1_sample=self.sample_h_given_v(v0_sample);
    pre_v1, v1_mean, v1_sample=self.sample_v_given_h(h1_sample);
    
    return [pre_h1, h1_mean, h1_sample,
            pre_v1, v1_mean, v1_sample];
            
  def free_energy(self, v_sample):
    
    wx_b=T.dot(v_sample, self.W)+self.h_bias;
    vbias_term=T.dot(v_sample, self.v_bias);
    hidden_term=T.sum(T.log(1+T.exp(wx_b)), axis=1);
    
    return -hidden_term-vbias_term;
  
  def get_updates(self,
                  learning_rate=0.1,
                  presistent=None,
                  k=1):
    
    pre_ph, ph_mean, ph_sample=self.sample_h_given_v(self.input);
    
    if presistent is None:
      chain_start=ph_sample;
    else:
      chain_start=presistent;
      
    ([pre_nvs,
      nv_means,
      nv_samples,
      pre_nhs,
      nh_means,
      nh_samples],
      updates)=theano.scan(self.gibbs_hvh,
                           outputs_info=[None, None, None, None, None, chain_start],
                           n_steps=k);
                               
    chain_end=nv_samples[-1];
    
    cost=T.mean(self.free_energy(self.input))-T.mean(self.free_energy(chain_end));
    
    gparams=T.grad(cost, self.params, consider_constant=[chain_end]);
    
    for gparam, param in zip(gparams, self.params):
      updates[param]=param-gparam*T.cast(learning_rate, dtype="float32");
      
    if presistent:
      updates[presistent]=nh_samples[-1];
      
      monitoring_cost=self.get_pseudo_cost(updates);
    else:
      monitoring_cost=self.get_reconstruction_cost(updates,
                                                   pre_nvs[-1]);
                                              
    return monitoring_cost, updates;
  
  def get_pseudo_cost(self, updates):
    
    bit_i_idx=theano.shared(value=0, name="bit_i_idx");
    
    xi=T.round(self.input);
    
    fe_xi=self.free_energy(xi);
    
    xi_flip=T.set_subtensor(xi[:, bit_i_idx],
                            1-xi[:, bit_i_idx]);
                            
    fe_xi_flip=self.free_energy(xi_flip);
    
    cost=T.mean(self.n_vis*T.log(self.activation(fe_xi_flip-fe_xi)));
    
    updates[bit_i_idx]=(bit_i_idx+1) % self.n_vis;
    
    return cost;
  
  def get_reconstruction_cost(self,
                              updates,
                              pre_nv):
    
    return T.nnet.binary_crossentropy(self.input, self.activation(pre_nv)).mean();