'''
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: Utility functions
'''

import os;
import gzip, cPickle;

import numpy as np;
import theano;
import theano.tensor as T;

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
  
  mean_image=get_mean_image(train_set[0]);
  
  def shared_dataset(data_xy, borrow=True):
        
    data_x, data_y = data_xy;
    data_x-=mean_image;
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype='float32'),
                             borrow=borrow);
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype='float32'),
                             borrow=borrow);
        
    return shared_x, T.cast(shared_y, 'int32');

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
  
  
  
  
  