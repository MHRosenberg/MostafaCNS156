import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(np.pi*x)

def gBar(x,a):
    return np.dot(a,x)

N = 100 ### what's N?
Xout = np.random.uniform(low=-1,high=1, size=(3,N))


