"""
Steepest decent methods
"""

import numpy as np
from quasinewton import QuasiNewtonOptimizer

class SteepestDecentOptimizer(QuasiNewtonOptimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # no hessian involvement
        self.update_hessian = False
        self.update_hessian_reason = None

        # gradient is always updated
        self.update_gradient = True
        self.update_gradient_reason = None

    def solve(self, hessian, gradient):
        return -gradient


    def minimize(self, objective, x0, gradient=None):
        n = len(x0)
        identity = np.eye(n)
        return super().minimize(
            objective, x0, gradient=gradient, hessian=identity
        )
