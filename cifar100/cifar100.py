from tensorflow.contrib.learn.python.learn.datasets import base
from DataSet import DataSet
from tensorflow.python.framework import dtypes
import scipy.io as sio

def unpickle(file):
    return cPickle.load(file)
    

def read_data_sets(data_dir):
    filename = "cifar-100-python.tar.gz"
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    local_file = base.maybe_download(TRAIN_DATA, data_dir, SOURCE_URL)
    
   
    print('Extracting', filename)
    train_images,train_labels =[],[]
    test_images,test_labels =[],[]
    with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        if "train" in bytestream.name:
           i,l = _get_data(bytestream)
           train_images.extend(i)
           train_labels.extend(l) 
        if "test" in bytestream.name:
            i,l = _get_data(bytestream) 
            test_images.extend(i)
            test_labels.extend(l)
             
    train_images.reshape((0,3,2,1))
    test_images.reshape((0,3,2,1))

    train = DataSet(train_images, train_labels,dtype=dtypes.uint8,depth=100)
    test = DataSet(test_images, test_labels,dtype=dtypes.uint8,depth=100)
    
    return base.Datasets(train=train, validation=None, test=test)



def _get_data(file):
    dict = unpickle(file)
    return dict.pop("data",None),dict.pop("batch_label",None)


