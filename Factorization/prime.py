'''
Created on Jun 17, 2016

@author: KatherineMJB
'''
import numpy as np

def shuffle_in_unison(a, b):
    # this method shuffles the two arrays according to the same randomized
    # index
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def sieve (nbits):
    n = 2**nbits
    ret=[]
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p]):
            if(p != 2):
                ret.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return ret

def generate_data(nbits):
    primes = sieve(nbits)
    primes = np.array(primes)
    np.random.shuffle(primes)
    prime1=primes[0:300]
    prime2=primes[-301:-1]
    data_x = []
    data_y = []
    for p1 in prime1:
        for p2 in prime2:
            data_y.append(convert(min(p1, p2),nbits))
            data_x.append(convert(p1*p2,nbits*2))
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    return shuffle_in_unison(data_x, data_y)
    
def convert(number,bits):
    # this method converts a number into an array containing
    # bit2 values, a bit upper bound must be specified
    ret=[]
    for i in range(0,bits):
        if number%2==0:
            ret.append(-1)
        else:
            ret.append(1)
        number=number/2
    return ret
