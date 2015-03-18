"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: testing MLP network
"""

import numpy as np;
import theano;
import theano.tensor as T;

import telauges.utils as utils;
from telauges.hidden_layer import HiddenLayer;
from telauges.hidden_layer import SoftmaxLayer;

n_epochs=1000;
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

layer_1=HiddenLayer(rng=rng,
                    data_in=X,
                    n_in=784,
                    n_out=300,
                    activate_mode="relu");
                                        
layer_2=SoftmaxLayer(rng=rng,
                     data_in=layer_1.output,
                     n_in=300,
                     n_out=10);
                     
params=layer_1.params+layer_2.params;

cost=layer_2.cost(y)+0.001*((layer_1.W**2).sum()+(layer_2.W**2).sum());

gparams=T.grad(cost, params);

updates=[(param_i, param_i-0.1*grad_i)
         for param_i, grad_i in zip(params, gparams)];
         

test_model = theano.function(inputs=[index],
                             outputs=layer_2.errors(y),
                             givens={X: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]});
    
validate_model = theano.function(inputs=[index],
                                 outputs=layer_2.errors(y),
                                 givens={X: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]});
    
train_model = theano.function(inputs=[index],
                              outputs=cost,
                              updates=updates,
                              givens={X: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size: (index + 1) * batch_size]});
                                      
print "[MESSAGE] The model is built";
print "[MESSAGE] Start training"       

validation_frequency = n_train_batches;

validation_record=np.zeros((n_epochs, 1));
test_record=np.zeros((n_epochs, 1));

epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  for minibatch_index in xrange(n_train_batches):
    mlp_minibatch_avg_cost = train_model(minibatch_index);
    iter = (epoch - 1) * n_train_batches + minibatch_index;
        
    if (iter + 1) % validation_frequency == 0:
      validation_losses = [validate_model(i) for i
                               in xrange(n_valid_batches)];
      validation_record[epoch-1] = np.mean(validation_losses);

      print 'MLP MODEL';
            
      print('epoch %i, minibatch %i/%i, validation error %f %%' %
            (epoch, minibatch_index + 1, n_train_batches, validation_record[epoch-1] * 100.));

      test_losses = [test_model(i) for i
                     in xrange(n_test_batches)];
      test_record[epoch-1] = np.mean(test_losses);
                
      print(('     epoch %i, minibatch %i/%i, test error %f %%') %
            (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.));
            
      