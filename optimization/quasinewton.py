"""
Quasi newton optimization framework
"""

import numpy as np

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

        # will be callable functions
        self.objective = None
        self.gradient = None
        self.hessian = None

        # other book keeping
        self.minimizer_list = None
        self.success = False

    def do_hessian_update(self):
        pass

    def do_gradient_update(self):
        pass

    def solve(self, hessian, gradient):
        return -gradient

    def check_success(self, gradient):
        self.success = np.linalg.norm(gradient) < self.gradient_tolerance


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
        decent_direction = self.solve(self.hessian, self.gradient)

        # determine step size
        if self.line_search is not None:
            step_size = self.line_search()
        else:
            step_size = 1

        # perform update
        new_point = self.get_minimizer() + step_size * decent_direction
        self.push_minimizer(new_point)


    def get_minimizer(self):
        return self.minimizer_list[-1]

    def push_minimizer(self, point):
        self.minimizer_list.append(point)

    def set_initial_guess(self, x0):
        self.minimizer_list = [x0]

    def minimize(self, objective, x0, gradient=None, hessian=None):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.set_initial_guess(x0)

        iter_count = 0
        while not self.success and iter_count < self.maximum_iterations:
            self.update_minimizer()
            self.check_success()
            iter_count += 1

        if iter_count == self.maximum_iterations:
            print("Warning maximum iterations reached!")
            if not self.success:
                raise ValueError("Iteration did not converge!")

        return self.get_minimizer()

            


        



