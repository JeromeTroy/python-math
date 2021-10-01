import numpy as np
from numba import jit
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "../optimization/")

import line_search as ls

@jit(nopython=True)
def f(x):
    return x[0]**2 + x[1]**2

@jit(nopython=True)
def g(x):
    return 2 * x

default_step_size = 2.59
des_dir = np.array([-1, -1])
loc = np.array([2, 2])
armijo = 0.01
wolfe = 0.5

max_iter = 10

step_size_backtrack = ls.execute_line_search(
    f, g, loc, des_dir, default_step_size, armijo, wolfe,
    max_iter=max_iter
)

print(step_size_backtrack)

wrap = lambda s: f(loc + s * des_dir)

s_values = np.linspace(0, default_step_size, 100)
y_values = np.array(list(map(wrap, list(s_values))))

plt.plot(s_values, y_values)
plt.plot(step_size_backtrack, wrap(step_size_backtrack), "*")
plt.xlabel("step size")
plt.ylabel("wrapped f")
plt.show()

step_size_wolfe = ls.execute_line_search(
    f, g, loc, des_dir, default_step_size, armijo, wolfe,
    max_iter=max_iter, method="wolfe"
)

plt.plot(s_values, y_values)
plt.plot(step_size_wolfe, wrap(step_size_wolfe), "*")
plt.xlabel("step size")
plt.ylabel("wrapped f")
plt.show()
