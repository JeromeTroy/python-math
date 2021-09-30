"""
Simple test cases for optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.insert(1, "../optimization/")

from steepest_decent import SteepestDecentOptimizer
from quasinewton import NewtonOptimizer


# case 1: quadratic 2d function
f = lambda x : np.linalg.norm(x)**2

# have explicit gradient
g = lambda x: 2 * x

# also explicit hessian
h = lambda x: 2 * np.diagflat(np.ones_like(x))

# plotting
x_plot = np.linspace(-1, 1, 100)
X_plot, Y_plot = np.meshgrid(x_plot, x_plot)

input_values = list(zip(list(X_plot.ravel()), list(Y_plot.ravel())))
func_values = np.array(list(map(f, input_values)))

Z_plot = np.reshape(func_values, X_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X_plot, Y_plot, Z_plot, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# minimization
print("Minimizing with gradient decent")

x0 = np.array([2, 1])

opt = SteepestDecentOptimizer()
x_opt = opt.minimize(f, x0, gradient=g)

print("Optimality reached")
print(x_opt)

print("path:")
print(opt.minimizer_list)

# minimization
print("Minimizing with Newton's method")

x0 = np.array([2, 1])

opt = NewtonOptimizer()
x_opt = opt.minimize(f, x0, gradient=g, hessian=h)

print("Optimality reached")
print(x_opt)

print("path:")
print(opt.minimizer_list)
