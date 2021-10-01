"""
Simple test cases for BFGS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

import sys
sys.path.insert(1, "../optimization/")

from bfgs import BFGS

@jit
def f(x):
    return x[0] ** 2 + x[1] ** 2

@jit
def g(x):
    return 2 * x

opt_args = {
    "line_search" : "back",
    "armijo" : 0.2,
}

opt = BFGS(**opt_args)

x0 = np.array([2, 2])
x_opt = opt.minimize(f, x0, gradient=g)

print("optimality")
print(x_opt)

print("Path")
print(opt.minimizer_list)
