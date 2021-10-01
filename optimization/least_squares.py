import numpy as np

import sys
sys.path.insert(1, "../utils/")

from abstract_newton_opt import AbstractNewtonLikeOptimizer
import fdjac
import line_search as ls

class GaussNewton(AbstractNewtonLikeOptimizer):

    def __init__(self, **kwargs):

        keys = kwargs.keys()
        if "tol" in keys:
            self.square_tolerance = kwargs["tol"]
        else:
            self.square_tolerance = 1e-6

        if "jacobian_update_reason" in keys:
            self.update_jacobian = False
            self.update_jacobian_reason = kwargs["jacobian_update_reason"]
        else:
            self.update_jacobian = True
            self.update_jacobian_reason = None

        if "fd_jac_step" in keys:
            self.fd_jac_step = kwargs["fd_jac_step"]
        else:
            self.fd_jac_step = 1e-6


        self.jacobian = None
        self.current_jacobian = 0
