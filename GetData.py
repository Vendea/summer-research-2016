from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from sys import path
from os import getcwd
p = getcwd()+"/cifar10"
path.append(p)
from cifar10 import read_data_sets as c10
svhn = c10("/tmp/data")

from sys import path
from os import getcwd
p = getcwd()+"/cifar100"
path.append(p)
from cifar100 import read_data_sets as c100
svhn = c100("/tmp/data")

from sys import path
from os import getcwd
p = getcwd()++"/svhn"
path.append(p)
from svhn import read_data_sets as sv
svhn = sv("/tmp/data")