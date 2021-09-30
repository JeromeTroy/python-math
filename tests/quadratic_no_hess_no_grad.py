"""
Simple test cases for optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

import sys
sys.path.insert(1, "../optimization/")

from steepest_decent import SteepestDecentOptimizer
from quasinewton import NewtonOptimizer


# case 1: quadratic 2d function
@jit
def f(x):
    return x[0]**2 + x[1]**2

# no provided gradient or hessian

# minimization
print("Minimizing with gradient decent")

x0 = np.array([2, 1])

steepest_args = {
    "fd_grad_step" : 1e-2,
    "default_step_size" : 0.1
}
opt = SteepestDecentOptimizer(**steepest_args)
x_opt = opt.minimize(f, x0)

print("Optimality reached")
print(x_opt)

print("path:")
print(opt.minimizer_list)

# minimization
print("Minimizing with Newton's method")

x0 = np.array([2, 1])

opt = NewtonOptimizer()
x_opt = opt.minimize(f, x0)

print("Optimality reached")
print(x_opt)

print("path:")
print(opt.minimizer_list)
