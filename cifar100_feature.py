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