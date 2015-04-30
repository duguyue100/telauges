'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Utility functions
'''

import os;
import gzip, cPickle;
import cPickle as pickle;

import numpy as np;
import matplotlib.pyplot as plt;
import theano;
import theano.tensor as T;
from theano.compile import ViewOp;

def load_CIFAR_100(filename,
                   file_type="train"):
  """
  Load CIFAR 100 data set
  """
  datadict=pickle.load(open(filename, "rb"));
  
  X=datadict["data"];
  Y=datadict["coarse_labels"];
  
  if file_type=="train":
    X=X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float");
  elif file_type=="test":
    X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
    
  Y=np.array(Y);
    
  return X, Y;

def load_CIFAR_batch(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    with open(filename, 'r') as f:
        datadict=pickle.load(f);
        
        X=datadict['data'];
        Y=datadict['labels'];
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
        Y=np.array(Y);
        
        return X, Y;
        
        
def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param ROOT: string of data folder
    @return: Xtr, Ytr: training data and labels
    @return: Xte, Yte: testing data and labels
    """
    
    xs=[];
    ys=[];
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ));
        X, Y=load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
        
    Xtr=np.concatenate(xs);
    Ytr=np.concatenate(ys);
    
    del X, Y;
    
    Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    
    return Xtr, Ytr, Xte, Yte;

def visualize_CIFAR(X_train,
                    y_train,
                    samples_per_class):
    """
    A visualize function for CIFAR 
    """
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    num_classes=len(classes);
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    
    plt.show();
    
def shared_dataset(data_xy, borrow=True):
        
  data_x, data_y = data_xy;
  #data_x-=mean_image;
  shared_x = theano.shared(np.asarray(data_x,
                                      dtype='float32'),
                           borrow=borrow);
  shared_y = theano.shared(np.asarray(data_y,
                                      dtype='float32'),
                           borrow=borrow);
        
  return shared_x, T.cast(shared_y, 'int32');

def load_mnist(dataset):
  """
  Load MNIST dataset
  
  @param dataset: string
  """
    
  # Download the MNIST dataset if it is not present
  data_dir, data_file = os.path.split(dataset);
  if data_dir == "" and not os.path.isfile(dataset):
    # Check if dataset is in the data directory.
    new_path = os.path.join(os.path.split(__file__)[0],
                            "..",
                            "data",
                            dataset);
    if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
      dataset = new_path;

  if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    import urllib;
    origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz');
    print 'Downloading data from %s' % origin;
    urllib.urlretrieve(origin, dataset);

  print '... loading data';

    # Load the dataset
  f = gzip.open(dataset, 'rb');
  train_set, valid_set, test_set = cPickle.load(f);
  f.close();
  
  #mean_image=get_mean_image(train_set[0]);

  test_set_x, test_set_y = shared_dataset(test_set);
  valid_set_x, valid_set_y = shared_dataset(valid_set);
  train_set_x, train_set_y = shared_dataset(train_set);

  rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
          (test_set_x, test_set_y)];
  return rval;

def standardlize_image(X):
  """
  
  """
  
def get_mean_image(X):
  """
  Get average image from training set
  
  @param X: image matrix, each row is a image vector (N x M)
  """
  
  return np.mean(X, axis=0);
  
def get_shared_matrix(name,
                      out_size,
                      in_size=None,
                      weight_type="none"):
  """
  create shared weights or weights
  
  @param out_size: output size (int)
  @param in_size: input size (int)
  @param weight_type: "none", "tanh" and "sigmoid"
  
  @return: a shared matrix with size of in_size x out_size
  """
  
  if in_size is not None:
    if weight_type=="tanh":
      lower_bound=-np.sqrt(6. / (in_size + out_size));
      upper_bound=np.sqrt(6. / (in_size + out_size));
    elif weight_type=="sigmoid":
      lower_bound=-4*np.sqrt(6. / (in_size + out_size));
      upper_bound=4*np.sqrt(6. / (in_size + out_size));
    elif weight_type=="none":
      lower_bound=0;
      upper_bound=1./(in_size+out_size);
  
  if in_size==None:
    return theano.shared(value=np.asarray(np.random.uniform(low=0,
                                                            high=1./out_size,
                                                            size=(out_size, )),
                                          dtype="float32"),
                         name=name,
                         borrow=True);
  else:
    return theano.shared(value=np.asarray(np.random.uniform(low=lower_bound,
                                                            high=upper_bound,
                                                            size=(in_size, out_size)),
                                          dtype="float32"),
                         name=name,
                         borrow=True);