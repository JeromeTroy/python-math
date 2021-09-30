"""
Quasi newton optimization framework
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve

import sys
sys.path.insert(1, "../utils/")

import fdjac

class QuasiNewtonOptimizer():

    def __init__(self, **kwargs):

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

        if "max_iters" in keys:
            self.maximum_iterations = kwargs["max_iters"]
        else:
            self.maximum_iterations = 100

        if "line_search" in keys:
            self.line_search = kwargs["line_search"]
        else:
            self.line_search = None

        if "fd_grad_step" in keys:
            self.fd_grad_step = kwargs["fd_grad_step"]
        else:
            self.fd_grad_step = 1e-6

        if "fd_hess_step" in keys:
            self.fd_hess_step = kwargs["fd_hess_step"]
        else:
            self.fd_hess_step = 1e-6

        if "default_step_size" in keys:
            self.default_step_size = kwargs["default_step_size"]
        else:
            self.default_step_size = 1

        # will be callable functions
        self.objective = None
        self.gradient = None
        self.hessian = None

        # current values
        self.current_objective = 0
        self.current_gradient = 0
        self.current_hessian = 0

        # other book keeping
        self.minimizer_list = None
        self.success = False

    def do_hessian_update(self):
        self.current_hessian = self.hessian(self.get_minimizer())

    def do_gradient_update(self):
        self.current_gradient = self.gradient(self.get_minimizer())

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
        decent_direction = self.solve(self.current_hessian,
                                        self.current_gradient)

        # determine step size
        if self.line_search is not None:
            step_size = self.line_search()
        else:
            step_size = self.default_step_size

        # perform update
        new_point = self.get_minimizer() + step_size * decent_direction
        self.push_minimizer(new_point)


    def get_minimizer(self):
        return self.minimizer_list[-1]

    def push_minimizer(self, point):
        self.minimizer_list.append(point)

    def set_initial_guess(self, x0):
        self.minimizer_list = [x0]

    def set_gradient(self, gradient):
        if gradient is not None:
            self.gradient = gradient

        else:
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
            decent_direction = -cho_solve((c, lower), gradient)
        except np.linalg.LinAlgError:
            # choleksy factorization failed, not SPD
            # has to be symmetric
            decent_direction = -solve(hessian, gradient, assume_a="sym")

        return decent_direction
