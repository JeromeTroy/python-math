import numpy as np
from scipy.linalg import cho_factor, cho_solve

import sys
sys.path.insert(1, "../utils/")

from least_squares import GaussNewton
import fdjac
import line_search as ls

class LevenbergMarquardt(GaussNewton):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        keys = kwargs.keys()
        if "reg_increase" in keys:
            self.reg_increase = kwargs["reg_increase"]
        else:
            self.reg_increase = 4

        if "reg_decrease" in keys:
            self.reg_decrease = kwargs["reg_decrease"]
        else:
            self.reg_decrease = 10

        if "reg_dominance" in keys:
            self.reg_dominance = kwargs["reg_dominance"]
        else:
            self.reg_dominance = 100

        if "use_identity" in keys:
            self.use_identity = kwargs["use_identity"]
        else:
            self.use_identity = True

        self.regularization = 20


    def build_matrix(self, jacobian):
        jTj = jacobian.T.conj() @ jacobian
        if self.use_identity:
            if self.regularization / np.linalg.det(jTj) > self.reg_dominance:
                # regularization dominated, just gradient descent
                A = None
            else:
                # not dominated
                A = jTj + self.regularization * np.eye(self.input_size)
        else:
            # use diagonal of jTj, scale independent
            if self.regularization > self.reg_dominance:
                A = None
            else:
                A = jTj + self.regularization * np.diag(jTj)
        return A

    def determine_if_should_update_jacobian(self, new_square_objective):
        super().determine_if_should_update_jacobian(new_square_objective)
        if not self.update_jacobian or new_square_objective < \
            np.linalg.norm(self.current_objective)**2:
            # all's fine and dandy in the world
            self.regularization /= self.reg_decrease
        else:
            # need to update jacobian and did not decrease
            # more regularization
            self.regularization *= self.reg_increase

    def solve(self, jacobian, objective):
        A = self.build_matrix(jacobian)

        if A is not None:
            (c, lower) = cho_factor(A)

            des_direction = -cho_solve((c, lower),
                                jacobian.T.conj() @ objective)
        else:
            des_direction = -jacobian.T.conj() @ objective
        return des_direction
