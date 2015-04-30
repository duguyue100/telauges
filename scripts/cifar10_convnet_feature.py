"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: CIFAR-10 convnet feature 
"""

import cPickle as pickle;

import numpy as np;
import theano
import theano.tensor as T;
import matplotlib.pyplot as plt;

import telauges.utils as utils;
from telauges.conv_net_layer import ConvNetLayer;
from telauges.hidden_layer import HiddenLayer;
from telauges.hidden_layer import SoftmaxLayer;

n_epochs=50;
training_portion=1;
batch_size=200;
nkerns=[50, 20];
training_portion=1;
rng=np.random.RandomState(23455);

Xtr, Ytr, Xte, Yte=utils.load_CIFAR10("/home/arlmaster/workspace/telauges/data/CIFAR10");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;
#Xtrain=np.hstack((Ytr[None].T, Xtrain))[0:10000];
#Xtrain=Xtrain[Xtrain[:,0].argsort()];

print Xtrain.shape;
print Xtest.shape;

#data_train=(Xtrain, Ytr);
#data_test=(Xtest, Yte);
#train_set_x=utils.shared_dataset(data_train);
#test_set_x=utils.shared_dataset(data_test);

train_set_x=theano.shared(np.asarray(Xtrain,
                                     dtype='float32'),
                          borrow=True);
train_set_y=theano.shared(np.asarray(Ytr,
                                     dtype='float32'),
                          borrow=True);
train_set_y=T.cast(train_set_y, dtype="int32");                          

test_set_x=theano.shared(np.asarray(Xtest,
                                    dtype='float32'),
                         borrow=True);
test_set_y=theano.shared(np.asarray(Yte,
                                    dtype='float32'),
                         borrow=True);
test_set_y=T.cast(test_set_y, dtype="int32");

n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion);
n_test_batches=test_set_x.get_value(borrow=True).shape[0];

print n_train_batches;
print n_test_batches;
    
n_train_batches /= batch_size; # number of train data batches
n_test_batches /= batch_size;  # number of test data batches

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"

index=T.lscalar(); # batch index

X=T.matrix('X');  # input data source
y=T.ivector('y'); # input data label

images=X.reshape((batch_size, 1, 32, 32))

# input size (32, 32), (7, 7)
layer_0=ConvNetLayer(rng=rng,
                     feature_maps=images,
                     feature_shape=(batch_size, 1, 32, 32),
                     filter_shape=(nkerns[0], 1, 7, 7),
                     pool=True,
                     activate_mode="relu");
                     
filters=layer_0.filters;

for i in xrange(nkerns[0]):
  plt.subplot(8, 7, i);
  plt.imshow(filters.get_value(borrow=True)[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')

plt.show();

# input size (13, 13), (4, 4)
layer_1=ConvNetLayer(rng=rng,
                     feature_maps=layer_0.out_feature_maps,
                     feature_shape=(batch_size, nkerns[0], 13, 13),
                     filter_shape=(nkerns[1], nkerns[0], 4, 4),
                     pool=True,
                     activate_mode="relu");
                     
# output size (5, 5)
layer_2=HiddenLayer(rng=rng,
                    data_in=layer_1.out_feature_maps.flatten(2),
                    n_in=nkerns[1]*25,
                    n_out=500);
                    
layer_3=SoftmaxLayer(rng=rng,
                     data_in=layer_2.output,
                     n_in=500,
                     n_out=10);
                     

params=layer_0.params+layer_1.params+layer_2.params+layer_3.params;

cost=layer_3.cost(y)+0.001*((layer_0.filters**2).sum()+(layer_1.filters**2).sum()+(layer_2.W**2).sum()+(layer_3.W**2).sum());

gparams=T.grad(cost, params);

updates=[(param_i, param_i-0.1*grad_i)
         for param_i, grad_i in zip(params, gparams)];
         

test_model = theano.function(inputs=[index],
                             outputs=layer_3.errors(y),
                             givens={X: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]});
    
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
      test_losses = [test_model(i) for i
                     in xrange(n_test_batches)];
      test_record[epoch-1] = np.mean(test_losses);
                
      print(('     epoch %i, minibatch %i/%i, test error %f %%') %
            (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.));
            

filters=layer_0.filters;

for i in xrange(nkerns[0]):
  plt.subplot(8, 7, i);
  plt.imshow(filters.get_value(borrow=True)[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')

plt.show();


## Prepare data
train_output_feature=theano.function(inputs=[index],
                                    outputs=layer_2.get_output(layer_1.get_output(layer_0.get_output(X.reshape((batch_size, 1, 32, 32)))).flatten(2)),
                                    givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});

train_feature=np.asarray([]);
for batch_index in xrange(n_train_batches):
  temp=train_output_feature(batch_index);
  
  if not train_feature.size:
    train_feature=temp;
  else:
    train_feature=np.vstack((train_feature, temp));
    
train_feature=np.hstack((train_set_y.eval()[None].T, train_feature));

print train_feature.shape;

train_feature_random=train_feature;
train_feature.view("float32, float32, float32").sort(order=["f1"], axis=0);

print train_feature.shape;
print "[MESSAGE] Writing training set to file"

pickle.dump(train_feature, open("cifar10_train_convnet_feature_500_ordered.pkl", "w"));
pickle.dump(train_feature_random, open("cifar10_train_convnet_feature_500_random.pkl", "w"));

print "[MESSAGE] Training set is prepared"

test_output_feature=theano.function(inputs=[index],
                                    outputs=layer_2.get_output(layer_1.get_output(layer_0.get_output(X.reshape((batch_size, 1, 32, 32)))).flatten(2)),
                                    givens={X: test_set_x[index * batch_size: (index + 1) * batch_size]});

test_feature=np.asarray([]);        
for batch_index in xrange(n_test_batches):
  temp=test_output_feature(batch_index);
  
  if not test_feature.size:
    test_feature=temp;
  else:
    test_feature=np.vstack((test_feature, temp));
    
test_feature=np.hstack((test_set_y.eval()[None].T, test_feature));

test_feature_random=test_feature;
test_feature.view("float32, float32, float32").sort(order=["f1"], axis=0);

print test_feature.shape;
print "[MESSAGE] Writing testing set to file"

pickle.dump(test_feature, open("cifar10_test_convnet_feature_500_ordered.pkl", "w"));
pickle.dump(test_feature_random, open("cifar10_test_convnet_feature_500_random.pkl", "w"));

print "[MESSAGE] Testing set is prepared"

