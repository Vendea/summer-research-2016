import flatten
import tensorflow as tf

'''import time


start=time.time()
flatten()
end=time.time()

def dot_product_flatten(mat_list1, mat_list2):
    (list1, dims) = flatten(mat_list1)
    (list2, _) = flatten(mat_list2)
    list3 = []
    for n,m in zip(list1, list2):
        list3.append(n*m)
    return unflatten((list3, dims))

def dot_product_no_flatten(mat_list1, mat_list2):
    for i in range(num_elems(mat_list1)):
'''

# returns the number of elements in a variable's n-dim matrix given its shape
def tuple_product(t):
    product = 1
    for i in t:
        product *= i
    return product

# number of individual trainable numbers
def num_elems(mat_list):
    num = 0
    for var in mat_list:
        num += tuple_product(var.shape)
    return num

# finds the tensorflow variable whose matrix contains the element at the specified index
def find_variable(index):
    pos = 0
    next = 0
    for var in tf.trainable_variables():
        next += tuple_product(var.shape)
        if index < next:
            return (var, index - pos)
        pos = next
    return (-1, -1)

def get_trainable_value(index):
    (var, index) = find_variable(index)
    assert index != -1 # Don't proceed if the variable hasn't been found
    return valueAt(index, var.eval(), var.rank)

def value_at(index, matrix, dims):
    if dims == 1:
        return matrix[index]
    elemsPerLayer = tuple_produc(dims[1:])
    return valueAt(index % elemsPerLayer, matrix[index // elemsPerLayer], dims - 1)

def set_trainable_value(index, new_value, sess):
    (var, index) = find_variable(index)
    if index != -1:
        matrix = var.eval()
        set_value(index, matrix, var.rank, new_value)
        sess.run(var.assign(matrix))

def set_value(index, matrix, dims, value):
    if dims == 1:
        matrix[index] = value
    elemsPerLayer = tuple_product(dims[1:])
    set_value(index % elemsPerLayer, matrix[index // elemsPerLayer], dims - 1, value)
