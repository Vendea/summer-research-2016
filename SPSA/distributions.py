from random import random

def bernoulli():
    return (int(random() * 2) - 0.5) * 2

def segmented_uniform():
    b = bernoulli()
    if b == -1:
        return random() / 2 - 1
    if b == 1:
        return random() / 2 + 0.5

def u_shaped():
    # following normalized x^2 distribution
    cube = (2 * random() - 1)
    if cube < 0:
        return cube ** (1.0 / 3)
    else:
        return -((-cube) ** (1.0 / 3))
