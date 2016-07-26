from tensorflow.contrib.learn.python.learn.datasets import base
import tarfile
from tensorflow.python.platform import gfile
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]
path.append(p)
from DataSet import DataSet
from tensorflow.python.framework import dtypes
import scipy.io as sio
import numpy as np
import cPickle

def unpickle(file):
    return cPickle.load(file)

def read_data_sets(data_dir):
    filename = "cifar-100-python.tar.gz"
    print("getting data")
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    local_file = base.maybe_download(TRAIN_DATA, data_dir, SOURCE_URL)
    
   
    print('Extracting', filename)
    train_images,train_labels =[],[]
    test_images,test_labels =[],[]
    with gfile.Open(data_dir+"/"+filename, 'rb') as f, tarfile.open(fileobj=f) as tar:
        for x in tar.getnames():
            if "data_batch" in x:
               i,l = _get_data(tar.extractfile(x))
               train_images.extend(i.reshape((i.shape[0],32,32,3)))
               train_labels.extend(l) 
            if "test_batch" in x:
                i,l = _get_data(tar.extractfile(x)) 
                test_images.extend(i.reshape((i.shape[0],32,32,3)))
                test_labels.extend(l)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train = DataSet(train_images, train_labels,dtype=dtypes.uint8,depth=100)
    test = DataSet(test_images, test_labels,dtype=dtypes.uint8,depth=100)
    
    return base.Datasets(train=train, validation=None, test=test)



def _get_data(file):
    dict = unpickle(file)
    return dict.pop("data",None),dict.pop("labels",None)


