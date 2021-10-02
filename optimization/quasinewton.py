"""
Quasi newton optimization framework

QuasiNewtonOptimizer is the base class for all Quasi-Newton
methods
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve

import sys
sys.path.insert(1, "../utils/")

from abstract_newton_opt import AbstractNewtonLikeOptimizer
import fdjac
import line_search as ls

class QuasiNewtonOptimizer(AbstractNewtonLikeOptimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # unpack many parameters, all the dials...
        keys = kwargs.keys()

        if "grad_tol" in keys:
            self.gradient_tolerance = kwargs["grad_tol"]
        else:
            self.gradient_tolerance = 1e-6

        if "hessian_update_reason" in keys:
            self.update_hessian = False
            self.update_hessian_reason = kwargs["hessian_update_reason"]
        else:
            self.update_hessian = True
            self.update_hessian_reason = None

        if "gradient_update_reason" in keys:
            self.update_gradient = False
            self.update_gradient_reason = kwargs["gradient_update_reason"]
        else:
            self.update_gradient = True
            self.update_gradient_reason = None

        if "fd_grad_step" in keys:
            self.fd_grad_step = kwargs["fd_grad_step"]
        else:
            self.fd_grad_step = 1e-6

        if "fd_hess_step" in keys:
            self.fd_hess_step = kwargs["fd_hess_step"]
        else:
            self.fd_hess_step = 1e-6

        # will be callable functions
        self.gradient = None
        self.hessian = None

        # current values
        self.current_gradient = 0
        self.current_hessian = 0

        self.current_position_update = 0

    def determine_if_should_update_hessian(self, new_objective):
        """
        Determine if the trust region for the previous hessian FD
        computation is outdated

        Input:
            new_objective : float
                new objective value, this is the only value which
                is not stored in the object

        Output:
            set's the value of self.update_hessian
        """

        if self.update_hessian_reason == "untrusted":
            true_drop = new_objective - self.current_objective
            # expected from quadratic approximation
            expected_drop = np.dot(
                self.current_gradient, self.current_position_update
            ) + 0.5 * np.dot(
                self.current_position_update,
                self.current_hessian @ self.current_position_update
            )
            ratio = expected_drop / true_drop

            self.update_hessian = (ratio < self.trust_ratio)
        elif self.update_hessian_reason is None:
            # do nothing, hessian is already set to update
            pass
        else:
            raise NotImplementedError("No other methods of hessian updating are implemented")

        # final check
        if np.linalg.norm(self.current_position_update) < \
            self.gradient_tolerance and self.update_hessian_reason is not None:
            self.update_hessian = True



    def do_hessian_update(self):
        self.current_hessian = self.hessian(self.get_minimizer())
        if self.update_hessian_reason is not None:
            # we reset the hessian, so we are probably good for a bit
            self.update_hessian = False

    def do_gradient_update(self):
        self.current_gradient = self.gradient(self.get_minimizer())

    def do_line_search(self):
        step_size = ls.execute_line_search(
            self.objective, self.gradient, self.get_minimizer(),
            self.current_descent_direction, self.default_step_size,
            self.armijo, self.wolfe,
            method=self.line_search, max_iter=self.max_iter_ls
        )
        return step_size

    def solve(self, hessian, gradient):
        return -gradient

    def check_success(self):
        self.success = (
            np.linalg.norm(self.current_gradient) < self.gradient_tolerance
        )

    def update_minimizer(self):
        """
        Perform one step of minimization loop
        """

        # check if update hessian and gradient
        if self.update_hessian:
            self.do_hessian_update()

        if self.update_gradient:
            self.do_gradient_update()

        # apply linear solver on hessian and gradient
        self.current_descent_direction = self.solve(self.current_hessian,
                                        self.current_gradient)

        # determine step size
        if self.line_search is not None:
            self.current_step_size = self.do_line_search()
        else:
            self.current_step_size = self.default_step_size

        # perform update
        self.current_position_update = self.current_step_size * \
                        self.current_descent_direction
        new_point = self.get_minimizer() + self.current_position_update

        if self.objective(new_point) >= self.current_objective:
            # we did not descend
            self.current_descent_direction *= -1
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
        new_objective = self.objective(self.get_minimizer())
        self.determine_if_should_update_hessian(new_objective)
        self.current_objective = new_objective


    def set_gradient(self, gradient):
        if gradient is not None:
            self.gradient = gradient

        else:
            # newton's method is gradient based, so we should always update
            # the gradient
            self.gradient = lambda x: fdjac.fdgrad(
                self.objective, x, step=self.fd_grad_step
            )



    def set_hessian(self, hessian):
        if hessian is not None:
            self.hessian = hessian
        else:
            self.hessian = lambda x: fdjac.fdhess(
                self.objective, x, step=self.fd_hess_step
            )
            # since we are using a finite difference approx,
            # we want to avoid recomputing, since it's expensive
            self.update_hessian_reason = "untrusted"


    def minimize(self, objective, x0, gradient=None, hessian=None):
        self.objective = objective
        self.set_gradient(gradient)
        self.set_hessian(hessian)

        self.set_initial_guess(x0)

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

class NewtonOptimizer(QuasiNewtonOptimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # always update hessian
        self.update_hessian = True
        self.update_hessian_reason = None

        # always update gradient
        self.update_gradient = True
        self.update_gradient_reason = None

    def solve(self, hessian, gradient):
        """
        Solve as a linear system
        """
        # expect hessian to be SPD

        try:
            (c, lower) = cho_factor(hessian)
            des_direction = -cho_solve((c, lower), gradient)
        except np.linalg.LinAlgError:
            # choleksy factorization failed, not SPD
            # has to be symmetric
            des_direction = -solve(hessian, gradient, assume_a="sym")


        return des_direction
