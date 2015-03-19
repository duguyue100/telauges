"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: test of ConvNet autoencoder
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

index=T.lscalar(); # batch index

X=T.matrix('X');  # input data source
y=T.ivector('y'); # input data label

images=X.reshape((batch_size, 1, 28, 28))

ae=ConvAE(rng=rng,
          feature_maps=images,
          feature_shape=(batch_size, 1, 28, 28),
          filter_shape=(nkerns, 1, 7, 7),
          encode_activate_mode="relu",
          decode_activate_mode="sigmoid");
          
cost, updates=ae.get_updates(learning_rate=0.1);

train_model = theano.function(inputs=[index],
                              outputs=cost,
                              updates=updates,
                              givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});
                                      
print "[MESSAGE] The model is built";
print "[MESSAGE] Start training"

filters=ae.encode_layer.filters.get_value(borrow=True);
for i in xrange(nkerns):
  plt.subplot(7, 7, i);
  plt.imshow(filters[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
plt.show();

epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  c = []
  for batch_index in xrange(n_train_batches):
    c.append(train_model(batch_index))

  print 'Training epoch %d, cost ' % epoch, np.mean(c);
  
  
filters=ae.encode_layer.filters.get_value(borrow=True);
for i in xrange(nkerns):
  plt.subplot(7, 7, i);
  plt.imshow(filters[i,0,:,:], cmap = plt.get_cmap('hot'), interpolation='nearest');
  plt.axis('off')
plt.show();
          
