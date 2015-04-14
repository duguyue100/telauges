"""
Extract CIFAR-100 feature from AutoEncoder
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
rng=np.random.RandomState(23455);

Xtr, Ytr=utils.load_CIFAR_100("/home/arlmaster/workspace/telauges/data/cifar-100-python/train");
Xte, Yte=utils.load_CIFAR_100("/home/arlmaster/workspace/telauges/data/cifar-100-python/test", file_type="test");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;

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

X=T.matrix("X");
y=T.ivector("y");
index=T.lscalar();

ae=AutoEncoder(rng=rng,
               data_in=X,
               n_vis=1024,
               n_hidden=500,
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
  plt.imshow(np.reshape(filters[:,i], (32, 32)), cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
plt.show();


epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  c = []
  for batch_index in xrange(n_train_batches):
    c.append(train_model(batch_index))

  print 'Training epoch %d, cost ' % epoch, np.mean(c);
  
#filters=ae.encode_layer.W.get_value(borrow=True);
#for i in xrange(100):
#  plt.subplot(10, 10, i);
#  plt.imshow(np.reshape(filters[:,i], (32, 32)), cmap = plt.get_cmap('gray'), interpolation='nearest');
#  plt.axis('off')
#plt.show();

## extract feature

train_output_feature=theano.function(inputs=[index],
                                     outputs=ae.get_feature(X),
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

#train_feature.view("float32, float32, float32").sort(order=["f1"], axis=0);

#valid_output_feature=theano.function(inputs=[index],
#                                     outputs=ae.get_feature(X),
#                                     givens={X: test_set_x[index * batch_size: (index + 1) * batch_size]});

#valid_feature=np.asarray([]);        
#for batch_index in xrange(n_valid_batches):
#  temp=valid_output_feature(batch_index);
#  
#  if not valid_feature.size:
#    valid_feature=temp;
#  else:
#    valid_feature=np.vstack((valid_feature, temp));
#    
#valid_feature=np.hstack((valid_set_y.eval()[None].T, valid_feature));

#train_feature=np.vstack((train_feature, valid_feature));
train_feature_random=train_feature;
train_feature.view("float32, float32, float32").sort(order=["f1"], axis=0);

print train_feature.shape;
print "[MESSAGE] Writing training set to file"

pickle.dump(train_feature, open("cifar100_train_feature_ordered.pkl", "w"));
pickle.dump(train_feature_random, open("cifar100_train_feature_random.pkl", "w"));

print "[MESSAGE] Training set is prepared"

test_output_feature=theano.function(inputs=[index],
                                    outputs=ae.get_feature(X),
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

pickle.dump(test_feature, open("cifar100_test_feature_ordered.pkl", "w"));
pickle.dump(test_feature_random, open("cifar100_test_feature_random.pkl", "w"));

print "[MESSAGE] Testing set is prepared"
