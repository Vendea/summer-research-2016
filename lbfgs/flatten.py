__author__ = 'billywu'


import numpy as np

def tuple_product(t):
    product = 1
    for i in t:
        product *= i
    return product

def flatten_mat(mat):
    if (mat.shape[0] == 0 or type(mat[0]) != type(np.array([]))):
        return mat
    else:
        new_list = []
        dim = mat.shape[0]
        for i in range(dim):
            for j in mat[i]:
                new_list.append(j)
        return flatten_mat(np.array(new_list))

def flatten(mat_list):
    flattened = []
    dims = []
    for mat in mat_list:
        dims.append((mat.shape, tuple_product(mat.shape)))
        for i in flatten_mat(mat):
            flattened.append(i)
    return (flattened, dims)

def unflatten(flattened_and_dims):
    (flattened, dims) = flattened_and_dims
    start = 0
    mat_list = []
    for (dim, num_elems) in dims:
        mat_list.append(np.array(flattened[start:start+num_elems]).reshape(dim))
        start += num_elems
    return mat_list
