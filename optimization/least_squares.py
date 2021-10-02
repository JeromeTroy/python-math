import numpy as np
from scipy.linalg import qr, solve, svd

import sys
sys.path.insert(1, "../utils/")

from abstract_newton_opt import AbstractNewtonLikeOptimizer
import fdjac
import line_search as ls

class GaussNewton(AbstractNewtonLikeOptimizer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

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

        self.previous_objective = None

        self.input_size = 0
        self.objective_size = 0

    def do_jacobian_update(self):
        self.current_jacobian = self.jacobian(self.get_minimizer())

    def determine_if_should_update_jacobian(self, new_square_objective):
        """
        Determine if the jacobian is outdated and needs to be updated

        Input:
            new_square_objective : float
                value of || f(x_{k+1}) ||^2
                this is the only value not accessable from self
        Output:
            sets the value of self.update_jacobian
        """

        if self.update_jacobian_reason == "untrusted":
            curr_square_objective = np.linalg.norm(self.current_objective)**2
            true_drop = new_square_objective - curr_square_objective
            expected_drop = np.linalg.norm(
                self.current_objective +
                self.current_jacobian @ self.current_position_update
            )**2 - curr_square_objective

            ratio = expected_drop / true_drop

            if true_drop >= 0 or ratio < self.trust_ratio:
                self.update_jacobian = True
            else:
                self.update_jacobian = False

        elif self.update_jacobian_reason is None:
            # do nothing
            pass
        else:
            raise NotImplementedError("No other methods of jacobian updating are implemented")

        # final check
        if np.linalg.norm(self.current_position_update) < \
            self.square_tolerance and self.update_jacobian_reason is not None:
            self.update_jacobian = True

    def square_objective(self, x):
        return np.linalg.norm(self.objective(x)) ** 2

    def do_line_search(self):
        ls_objective = lambda x: self.current_jacobian.T.conj() @ \
                self.objective(x)
        step_size = ls.execute_line_search(
            self.square_objective, ls_objective, self.get_minimizer(),
            self.current_descent_direction, self.default_step_size,
            self.armijo, self.wolfe,
            method=self.line_search, max_iter=self.max_iter_ls
        )
        return step_size

    def solve(self, jacbian, objective):

        # J is m x n with m > n, over constrained
        Q, R = qr(jacbian, mode="economic")
        tmp = Q.T @ objective
        des_direction = -solve(R, tmp)


        return des_direction

    def check_success(self):
        self.success = (
            self.square_objective(self.get_minimizer()) < self.square_tolerance
        )

    def update_minimizer(self):

        # check for update needed
        if self.update_jacobian:
            self.do_jacobian_update()
        elif self.previous_objective is not None and \
            len(self.minimizer_list) >= 2:
            # broyden update on jacobian
            y = self.current_objective - self.previous_objective
            s = self.get_minimizer() - self.minimizer_list[-2]
            broyden_update = np.outer(
                y - self.current_jacobian @ s, s) / np.dot(s, s)

            self.current_jacobian += broyden_update

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
        new_square_objective = self.square_objective(new_point)
        self.determine_if_should_update_jacobian(new_square_objective)

        self.push_minimizer(new_point)
        self.previous_objective = self.current_objective
        self.current_objective = self.objective(self.get_minimizer())

    def set_jacobian(self, jac):
        if jac is not None:
            self.jacobian = jac

        else:
            self.jacobian = lambda x: fdjac.fdjacobian(
                self.objective, x, step=self.fd_jac_step
            )
            # since the fd computation for the jacobian is expensive,
            # we want to avoid recomputing if we don't have to
            self.update_jacobian_reason = "untrusted"

    def set_initial_guess(self, x0):
        super().set_initial_guess(x0)
        self.input_size = x0.shape[0]
        self.objective_size = self.objective(x0).shape[0]

    def minimize(self, objective, x0, jacobian=None, data=None):
        self.input_size = x0.shape[0]
        if data is not None:
            new_objective = lambda x: objective(x) - data
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
