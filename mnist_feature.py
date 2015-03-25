"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Extract MNIST feature
"""

import cPickle as pickle;

import numpy as np;
import theano;
import theano.tensor as T;
import matplotlib.pyplot as plt;

import telauges.utils as utils;
from telauges.hidden_layer import AutoEncoder;

n_epochs=100;
training_portion=1;
batch_size=100;

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

X=T.matrix("X");
y=T.ivector("y");
index=T.lscalar();

ae=AutoEncoder(rng=rng,
               data_in=X,
               n_vis=784,
               n_hidden=200,
               encode_activate_mode="sigmoid",
               decode_activate_mode="sigmoid");
               

    
cost, updates=ae.get_updates(learning_rate=0.1,
                             corruption_level=0.3);

    
train_model = theano.function(inputs=[index],
                              outputs=cost,
                              updates=updates,
                              givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});
                                      
print "[MESSAGE] The model is built";
print "[MESSAGE] Start training"

filters=ae.encode_layer.W.get_value(borrow=True);

for i in xrange(100):
  plt.subplot(10, 10, i);
  plt.imshow(np.reshape(filters[:,i], (28, 28)), cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
plt.show();


epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  c = []
  for batch_index in xrange(n_train_batches):
    c.append(train_model(batch_index))

  print 'Training epoch %d, cost ' % epoch, np.mean(c);
  
filters=ae.encode_layer.W.get_value(borrow=True);
for i in xrange(100):
  plt.subplot(10, 10, i);
  plt.imshow(np.reshape(filters[:,i], (28, 28)), cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
plt.show();


output_feature=theano.function(inputs=[index],
                               outputs=ae.get_feature(X),
                               givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});
                               
feature=np.asarray([]);
for batch_index in xrange(n_train_batches):
  temp=output_feature(batch_index);
  
  if not feature.size:
    feature=temp;
  else:
    feature=np.vstack((feature, temp));
    
feature=np.hstack((train_set_y.eval()[None].T, feature));

print feature.shape;

feature.view("float32, float32, float32").sort(order=["f1"], axis=0);

pickle.dump(feature, open("mnist_feature.pkl", "w"));