from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import scipy.io as sio

def read_data_sets(data_dir):
    TRAIN_DATA = 'train_32x32.mat'
    TEST_DATA = 'test_32x32.mat'

    SOURCE_URL = 'http://ufldl.stanford.edu/housenumbers/'

    local_file = base.maybe_download(TRAIN_DATA, data_dir, SOURCE_URL + TRAIN_DATA)
    train_data = sio.loadmat(local_file)
    train_images = train_data['X'].transpose(3,0,1,2)
    train_labels = train_data['y']

    local_file = base.maybe_download(TEST_DATA, data_dir, SOURCE_URL + TRAIN_DATA)
    test_data = sio.loadmat(local_file)
    test_images = test_data['X'].transpose(3,0,1,2)
    test_labels = test_data['y']

    train = DataSet(train_images, train_labels, dtype=dtypes.uint8)
    test = DataSet(test_images, test_labels, dtype=dtypes.uint8)

    return base.Datasets(train=train, validation=None, test=test)
