import numpy as np
import numba

import sys
sys.path.insert(1, "../utils/")

import fdjac


# case 1: quadratic 2d function
@numba.jit
def f(x):
    return np.linalg.norm(x)**2

# have explicit gradient
g = lambda x: 2 * x

# also explicit hessian
h = lambda x: 2 * np.diagflat(np.ones_like(x))

g_approx = lambda x: fdjac.fdgrad(f, x)
h_approx = lambda x: fdjac.fdhess(f, x)

origin = np.zeros(2, float)
point = np.ones(2, float)

diff_at_zero = g_approx(origin) - g(origin)
print("gradient difference at zero")
print(diff_at_zero)

diff_at_one_one = g_approx(point) - g(point)
print("gradient difference at (1, 1)")
print(diff_at_one_one)

diff_at_zero = h_approx(origin) - h(origin)
print("hessian difference at zero")
print(diff_at_zero)

diff_at_one_one = h_approx(point) - h(point)
print("hessian difference at (1, 1)")
print(diff_at_one_one)
