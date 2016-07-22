from tensorflow.contrib.learn.python.learn.datasets import base
import gzip
from tensorflow.python.platform import gfile
from sys import path
from os import getcwd
p = getcwd()[0:getcwd().rfind("/")]
path.append(p)
from DataSet import DataSet
from tensorflow.python.framework import dtypes
import scipy.io as sio
import numpy as np
def unpickle(file):
    return cPickle.load(file)

def read_data_sets(data_dir):
    filename = "cifar-10-python.tar.gz"
    print("getting data")
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    local_file = base.maybe_download(filename, data_dir, SOURCE_URL)
    
   
    print('Extracting', filename)
    train_images,train_labels =[],[]
    test_images,test_labels =[],[]
    with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        print(bytestream.name)
        if "data" in bytestream.name:
           i,l = _get_data(bytestream)
           print(i)
           train_images.extend(i.reshape((0,3,2,1)))
           train_labels.extend(l) 
        if "test" in bytestream.name:
            i,l = _get_data(bytestream) 
            test_images.extend(i.reshape((0,3,2,1)))
            test_labels.extend(l)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train = DataSet(train_images, train_labels,dtype=dtypes.uint8,depth=10)
    test = DataSet(test_images, test_labels,dtype=dtypes.uint8,depth=10)
    
    return base.Datasets(train=train, validation=None, test=test)



def _get_data(file):
    dict = unpickle(file)
    return dict.pop("data",None),dict.pop("batch_label",None)


