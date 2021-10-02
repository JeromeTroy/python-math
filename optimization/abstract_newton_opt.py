"""
Abstract class for newton-like optimizer

Base class for both minimization and least squares methods
"""
class AbstractNewtonLikeOptimizer():

    def __init__(self, **kwargs):

        keys = kwargs.keys()

        if "max_iters" in keys:
            self.maximum_iterations = kwargs["max_iters"]
        else:
            self.maximum_iterations = 100

        if "line_search" in keys:
            self.line_search = kwargs["line_search"]
        else:
            self.line_search = None

        if "default_step_size" in keys:
            self.default_step_size = kwargs["default_step_size"]
        else:
            self.default_step_size = 1

        if "armijo" in keys:
            self.armijo = kwargs["armijo"]
        else:
            self.armijo = 0.5
        if "wolfe" in keys:
            self.wolfe = kwargs["wolfe"]
        else:
            self.wolfe = 0.3

        if "line_search_max_iter" in keys:
            self.max_iter_ls = kwargs["line_search_max_iter"]
        else:
            self.max_iter_ls = 15

        if "fd_trust" in keys:
            self.trust_ratio = kwargs["fd_trust"]
        else:
            self.trust_ratio = 0.25


        # variable allocation
        self.objective = None

        self.current_objective = None

        self.current_step_size = self.default_step_size
        self.current_descent_direction = None

        # other book keeping
        self.minimizer_list = None
        self.success = False

    def get_minimizer(self):
        return self.minimizer_list[-1]

    def push_minimizer(self, point):
        self.minimizer_list.append(point)

    def set_initial_guess(self, x0):
        self.minimizer_list = [x0]
        self.current_objective = self.objective(x0)
