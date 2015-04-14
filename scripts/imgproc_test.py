"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Test on image processing functions
"""

import numpy as np;
import theano
import theano.tensor as T;
import matplotlib.pyplot as plt;

import telauges.utils as utils;
from telauges.conv_ae import ConvAE;

n_epochs=50;
training_portion=1;
batch_size=20;
nkerns=49;

datasets=utils.load_mnist("data/mnist.pkl.gz");
rng=np.random.RandomState(23455);

### Loading and preparing dataset
train_set_x, train_set_y = datasets[0];
valid_set_x, valid_set_y = datasets[1];
test_set_x, test_set_y = datasets[2];
    
n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion);
n_valid_batches=valid_set_x.get_value(borrow=True).shape[0];
n_test_batches=test_set_x.get_value(borrow=True).shape[0];
    
n_train_batches /= batch_size; # number of train data batches
n_valid_batches /= batch_size; # number of valid data batches
n_test_batches /= batch_size;  # number of test data batches

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"

print np.mean(test_set_x.get_value(borrow=True));