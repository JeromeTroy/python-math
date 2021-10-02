import numpy as np
from scipy.linalg import qr, solve

import sys
sys.path.insert(1, "../optimization/")

from least_squares import GaussNewton
from levenberg import LevenbergMarquardt

m = 5
n = 3

A = np.random.random((m, n))
x0 = np.random.random(n)
y = np.random.random(m)

# goal: solve A x = 0 using least squares
f = lambda x: A @ x
jac = lambda x: A

opt_args = {
    "line_search" : "back",
    "default_step_size" : 0.5,
    "armijo" : 0.2,
    "max_iters" : 50,
    "tol" : 1e-3
}
opt = GaussNewton(**opt_args)
x_opt = opt.minimize(f, x0, data=y)
print("optimum", x_opt)

Q, R = qr(A, mode="economic")
x_lin = solve(R, Q.T @ y)
diff = x_opt - x_lin
print("difference", diff)
print("relative difference", diff / x_lin)

new_opt = LevenbergMarquardt(**opt_args)
x_opt_2 = new_opt.minimize(f, x0, data=y)

print("optimum", x_opt_2)
