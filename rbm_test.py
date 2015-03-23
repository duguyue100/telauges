"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Test RBM
"""

import numpy as np;
import theano;
import theano.tensor as T;
import matplotlib.pyplot as plt;

import telauges.utils as utils;
from telauges.rbm import RBM;

n_epochs=15;
training_portion=1;
batch_size=20;

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

n_hidden=300;
presistent_chain=theano.shared(np.zeros((batch_size, n_hidden),
                                        dtype="float32"),
                               borrow=True);
                               
rbm=RBM(rng=rng,
        data_in=X,
        n_vis=28*28,
        n_hidden=n_hidden,
        activate_mode="sigmoid");
        
cost, updates=rbm.get_updates(learning_rate=0.1, presistent=presistent_chain, k=15);

train_rbm=theano.function([index],
                          cost,
                          updates=updates,
                          givens={X: train_set_x[index*batch_size:(index+1)*batch_size]});
                          
print "[MESSAGE] Start Training"

for epoch in xrange(n_epochs):
  mean_cost=[];
  
  for batch_id in xrange(n_train_batches):
    mean_cost+=[train_rbm(batch_id)];
    
  print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost);


filters=rbm.W.get_value(borrow=True);
for i in xrange(100):
  plt.subplot(10, 10, i);
  plt.imshow(np.reshape(filters[:,i], (28, 28)), cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
plt.show();







