"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: CKM tests
"""

import numpy as np;
import theano
import theano.tensor as T;
from scipy.misc import imread, imresize;
import matplotlib.pyplot as plt;

import telauges.utils as utils;
from telauges.ckm_layer import CKMLayer;

n_epochs=30;
training_portion=0.2;
batch_size=1;
nkerns=[56, 20];
rng=np.random.RandomState(125);

Xtr, Ytr, Xte, Yte=utils.load_CIFAR10("/home/arlmonster/workspace/telauges/data/CIFAR10");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;

from sklearn import preprocessing
Xtrain=preprocessing.scale(Xtrain);

img=imread("test1.JPEG");

img=np.array(img, dtype=float);
img=np.mean(img, axis=2)/255.0;
wid=img.shape[0];
height=img.shape[1];
img=np.ndarray.flatten(img)[None];
#img=preprocessing.scale(img);

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
    
n_train_batches /= batch_size; # number of train data batches
n_test_batches /= batch_size;  # number of test data batches

n_train_batches=5;

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"


index=T.lscalar(); # batch index

X=T.matrix('X');  # input data source
y=T.ivector('y'); # input data label

images=X.reshape((batch_size, 1, 32, 32))

layer=CKMLayer(rng=rng,
               feature_maps=images,
               feature_shape=(batch_size, 1, 32, 32),
               filter_shape=(nkerns[0], 1, 8, 8),
               pool=True,
               activate_mode="relu")

updates, cost, update_out=layer.ckm_updates();

train_model=theano.function(inputs=[index],
                            outputs=[cost, update_out],
                            updates=updates,
                            givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});
                            
print "... model is built"

epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  c = []
  for batch_index in xrange(n_train_batches):
    co, update_out=train_model(batch_index)
    c.append(co)
    
    #print update_out
    #print np.mean(update_out, axis=(1,2,3), keepdims=True)
    #print np.max(update_out, axis=(1,2,3), keepdims=True)
    #print np.array(update_out>=np.mean(update_out, axis=(2,3), keepdims=True), dtype=float);
    #plt.imshow(update_out[0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest')
    #plt.show()
    
#     for i in xrange(nkerns[0]):
#       plt.subplot(8, 7, i);
#       plt.imshow(update_out[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
#       plt.axis('off')
#     plt.show();
    
  print 'Training epoch %d, cost ' % epoch, np.mean(c);
  
filters=layer.filters;

for i in xrange(nkerns[0]):
  plt.subplot(8, 7, i);
  plt.imshow(filters.get_value(borrow=True)[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')

plt.show();