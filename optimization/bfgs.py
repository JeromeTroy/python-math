"""
BFGS and friends
"""

import numpy as np
from quasinewton import QuasiNewtonOptimizer
from scipy.linalg import cho_factor, cho_solve

class BFGS(QuasiNewtonOptimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # hessian will always be updated
        self.update_hessian = True
        self.update_hessian_reason = None

        # gradient is also always updated
        self.update_gradient = True
        self.update_gradient_reason = None

        # need to record previous values
        self.previous_gradient = None


    def do_hessian_update(self):
        # apply bfgs update to hessian
        if self.previous_gradient is None:
            # initial case
            self.current_hessian = np.eye(len(self.get_minimizer()))

        else:
            # bfgs update
            y = self.current_gradient - self.previous_gradient
            s = self.current_step_size * self.current_position_update

            z = self.current_hessian @ s
            hessian_update = np.outer(y, y) / np.dot(y, s) - \
                np.outer(z, z) / np.dot(s, z)
            self.current_hessian += hessian_update

    def do_gradient_update(self):
        # store previous gradient first
        self.previous_gradient = self.current_gradient
        super().do_gradient_update()
