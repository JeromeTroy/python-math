import numpy as np
from scipy.linalg import qr, solve, svd

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

        self.input_size = 0
        self.objective_size = 0

    def do_jacobian_update(self):
        self.current_jacobian = self.jacobian(self.get_minimizer())

    def square_objective(self, x):
        return np.linalg.norm(self.objective(x)) ** 2

    def do_line_search(self):
        step_size = ls.execute_line_search(
            self.square_objective, self.objective, self.get_minimizer(),
            self.current_descent_direction, self.default_step_size,
            self.armijo, self.wolfe,
            method=self.line_search, max_iter=self.max_iter_ls
        )
        return step_size

    def solve(self, jacbian, objective):

        # J is m x n with m > n, over constrained
        Q, R = qr(jacbian, mode="economic")
        tmp = Q.T @ objective
        des_direction = solve(R, tmp)


        return des_direction

    def check_success(self):
        self.success = (
            self.square_objective(self.get_minimizer()) < self.square_tolerance
        )

    def update_minimizer(self):

        # check for update needed
        if self.update_jacobian:
            self.do_jacobian_update()

        # determine descent direction
        self.current_descent_direction = self.solve(self.current_jacobian,
                                                    self.current_objective)

        # determine step size
        if self.line_search is not None:
            self.current_step_size = self.do_line_search()
        else:
            self.current_step_size = self.default_step_size

        # perform update
        self.current_position_update = self.current_step_size * \
                        self.current_descent_direction
        new_point = self.get_minimizer() + self.current_position_update

        self.push_minimizer(new_point)
        self.current_objective = self.objective(self.get_minimizer())

    def set_jacobian(self, jac):
        if jac is not None:
            self.jacobian = jac

        else:
            self.jac = lambda x: fdjac.fdjacobian(
                self.objective, x, step=self.fd_jac_step
            )

    def set_initial_guess(self, x0):
        super().set_initial_guess(x0)
        self.input_size = x0.shape[0]
        self.objective_size = self.objective(x0).shape[0]

    def minimize(self, objective, x0, jacobian=None, data=None):
        if data is not None:
            new_objective = lambda x: self.objective - data
            self.objective = new_objective
        else:
            self.objective = objective

        self.set_jacobian(jacobian)

        self.set_initial_guess(x0)

        # main loop
        iter_count = 0
        while not self.success and iter_count < self.maximum_iterations:
            self.update_minimizer()
            self.check_success()
            iter_count += 1

        if iter_count == self.maximum_iterations:
            print("Warning maximum iterations reached!")
            if not self.success:
                print("Minimization list: ")
                print(self.minimizer_list)
                print("Iteration did not converge!")

        return self.get_minimizer()
