"""
Line search methods for optimization
"""
import numpy as np
from numba import jit

def execute_line_search(fun, grad, loc, des_dir, default_step_size,
                        armijo, wolfe, method="back", max_iter=15):
    """
    Wrapper for executing line search

    Input:
        fun : callable, signature fun(x)
            objective function
        grad : callable, signature grad(x)
            gradient of objective
        loc : array of floats
            current location
        des_dir : array of floats
            proposed descent direction
        default_step_size : float
        armijo : float
            armijo condition constant
        wolfe : float
            wolfe condition constant
        method : string, optional
            line search method to use
            The default is backtracking
        max_iter : int, optional
            maximum number of line search iterations
            The default is 15
    Output:
        step : float
            found step size
    """
    step = default_step_size

    if method[0] not in ["b", "w"]:
        print("Unknown line search request, defaulting to None")

    elif method[0] == "b":
        # backtracking
        grad_val = grad(loc)
        step = backtracking(fun, grad_val, loc, des_dir,
            default_step_size, armijo, max_iters=max_iter)

    else:
        # wolfe search
        # raise NotImplementedError("Wolfe condition still a work in progress")
        step = wolfe_search(fun, grad, loc, des_dir,
                            default_step_size, max_iter, wolfe, armijo)

    return step



@jit
def check_armijo(fun_new, fun_old, dec_dot_grad, armijo):
    """
    Check if the armijo condition is satisfied:
        f(x + s p) <= f(x) + a * s p^T grad[f](x)

    Input:
        fun_new : float, f(x + s p)
        fun_old : float, f(x)
        dec_dot_grad : float, p^T grad[f](x)
        armijo : float, armijo constant (a)
    Output:
        check : boolean, True if condition is satisfied
    """

    check = (fun_new <= fun_old + armijo * dec_dot_grad)
    return check

@jit
def check_wolfe(new_pdg, old_pdg, wolfe):
    """
    Check the wolfe condition:
        |p^T grad[f](x + s p )| <= w * |p^T grad[f](x)|

    Input:
        new_pdg : float
            value of p^T grad[f](x + s p)
        old_pdg : float
            value of p^T grad[f](x)
        wolfe : float
            wolfe constant (w)
    Output:
        check : boolean, True if condition is satisfied
    """

    check = np.abs(new_pdg) <= wolfe * np.abs(old_pdg)
    return check

def backtracking(fun, grad, loc, des_dir, default_step_size, armijo,
                    max_iters=15):
    """
    Backtracking line search

    Input:
        fun : callable, signature fun(x)
            objective function
        grad : array of floats
            gradient value at location
        loc : array of floats
            location of current guess for optimum
        des_dir : array of floats
            descent direction computed
        default_step_size : float
        armijo : float
            constant for armijo condition:
                f(x + s p) <= f(x) + a * s p^T grad[f](x)
        max_iters : int, optional
            maximum iterations for line search.
            The default is 15
    Output:
        new_step_size : float
            step size from backtracking
    """

    # first iteration will remote the "2 * "
    new_step_size = 2 * default_step_size
    ls_iters = 0

    fun_old = fun(loc)

    # pre do the dot product
    pdg = np.dot(grad, des_dir)
    is_suff_descent = False

    while not is_suff_descent and ls_iters < max_iters:
        # update step size
        new_step_size /= 2

        # check armijo
        fun_new = fun(loc + new_step_size * des_dir)
        is_suff_descent = check_armijo(fun_new, fun_old, pdg, armijo)

        ls_iters += 1

    if ls_iters == max_iters:
        print("Warning, line search could not satisfy Armijo Condition")

    return new_step_size

def bracketing(grad, loc, des_dir, default_step_size, max_iter, wolfe,
            debug=False):
    """
    Bracketing to determine interval of valid solution

    Input:
        grad : callable, signature grad(x)
            gradient of objective function
        loc : array of floats
            location of minimum estimate
        des_dir : array of floats
            proposed descent direction
        default_step_size : float
        max_iter : int
            maximum number of allowed iterations
        wolfe : float
            wolfe condition constant
    Output:
        bracket : [a, b], both floats
            bracket found so that optimal step size
            is in [a, b]
    """

    # setup bracket
    bracket = [default_step_size, 2 * default_step_size]
    bracket_success = False
    lower_success = False
    upper_success = False
    key_val = wolfe * np.abs(np.dot(des_dir, grad(loc)))

    # gradients at each end
    lower_grad = np.dot(des_dir, grad(loc + bracket[0] * des_dir))
    upper_grad = np.dot(des_dir, grad(loc + bracket[1] * des_dir))

    iter_num = 0

    while not bracket_success and iter_num < max_iter:
        iter_num += 1
        # what can be wrong

        if lower_grad > 0:
            if debug: print("lower is too far out")
            bracket[0] = 0
            lower_grad = np.dot(des_dir, grad(loc))
            continue

        elif np.abs(lower_grad) >= key_val:
            if debug: print("lower is not far out enough")

            if upper_grad < 0 and np.abs(upper_grad) < key_val:
                if debug: print("can replace with upper")
                bracket[0] = bracket[1]
                bracket[1] *= 2
                lower_grad = upper_grad
                upper_grad = np.dot(des_dir, grad(loc + bracket[1] * des_dir))

            else:
                if debug: print("extending lower, but not to outer")
                bracket[0] = (7 * bracket[0] + 3 * bracket[1]) / 10
                lower_grad = np.dot(des_dir, grad(loc + bracket[0] * des_dir))

            continue

        else:
            if debug:
                print("lower grad is negative, and is within appropriate bounds")
            lower_success = True

        if upper_grad < 0:
            if debug: print("upper is not out far enough")
            bracket[1] *= 2
            upper_grad = np.dot(des_dir, grad(loc + bracket[1] * des_dir))
            continue

        elif np.abs(upper_grad) >= key_val:
            if debug: print("upper is too far out")
            bracket[1] = (3 * bracket[0] + 7 * bracket[1]) / 10
            upper_grad = np.dot(des_dir, grad(loc + bracket[1] * des_dir))
            continue

        else:
            if debug:
                print("upper grad is positive and is within appropriate bounds")
            upper_success = True

        bracket_success = lower_success and upper_success
        if debug: print("have success? ", bracket_success)

    bracket_success = lower_success and upper_success
    if not bracket_success:
        print("Warning wolfe condition cannot be satisfied")

    return bracket


#@jit
def __find_cubic_interpolant_root__(bracket, f_hi, g_hi, f_lo, g_lo):
    """
    Helper function for finding a cubic interpolant to find its minimum

    Input:
        bracket : [a, b]
        f_hi, g_hi : floats, function and gradient value at b
        f_lo, g_lo : floats, function and gradient value at a
    Output:
        root : location of minimum of cubic
    """

    # cubic of for A (x - a)^3 + B (x - b)^2 + C (x - b) + D
    # assign coefficients
    d = f_lo
    c = g_lo

    # width of bracket
    width = bracket[1] - bracket[0]

    # more complex coefficients
    b = -(g_hi * width - 3 * f_hi + 2 * c * width + 3 * d) / (width**2)
    a = (f_hi - b * width ** 2 - c * width - d) / (width ** 3)

    # check for real root via discriminant (in parabola from derivative)
    discriminant = b**2 - 3 * a * c
    opt_value = bracket[0]
    if discriminant > 0:
        # have at most 2 real roots
        if np.abs(a) > 1e-16:
            # cubic interpolant
            root_plus = (-b + np.sqrt(discriminant)) / (3 * a) + bracket[0]
            root_minus = (-b - np.sqrt(discriminant)) / (3 * a) + bracket[0]
        else:
            # is really quadratic
            root_plus = -c / (2 * b) + bracket[0]
            root_minus = root_plus

        if bracket[0] < root_plus and root_plus < bracket[1]:
            # + root is valid
            opt_value = root_plus
        elif bracket[0] < root_minus and root_minus < bracket[1]:
            # + root invalid, - root valid
            opt_value = root_minus

    else:
        print("Warning, cubic does not have valid root!")
    return opt_value

def wolfe_search(f, g, loc, des_dir,
                    default_step_size, max_iter, wolfe, armijo):
    """
    Wrapper to execute wolfe search

    Input:
        f : callable, signature f(x)
            objective function
        g : callable, signature g(x)
            gradient of objective function
        des_dir : array of floats
            current descent direction
        default_step_size : float
        max_iter : int
            maximum number of iterations for both wolfe bracketing
            and for wolfe search from cubic interval refinement
        wolfe : float
            wolfe condition constant
        armijo : float
            armijo condition constant

    Output:
        step_size : float
            computed step size
    """

    # stage one - compute bracket
    bracket = bracketing(g, loc, des_dir, default_step_size, max_iter, wolfe)

    # check for success for wolfe condition
    step_size = bracket[0]
    # store function and gradient values
    f_lo = f(loc + step_size * des_dir)
    g_lo = np.dot(des_dir, g(loc + step_size * des_dir))
    f_curr = f(loc)
    g_curr = g(loc)
    pdg_curr = np.dot(des_dir, g_curr)
    # check both wolfe condition and armijo condition
    wolfe_success = check_wolfe(pdg_curr, g_lo,
                                wolfe)
    wolfe_success = wolfe_success and \
            check_armijo(f_curr, f_lo, pdg_curr, armijo)

    if not wolfe_success:
        # check other side
        step_size = bracket[1]
        f_hi = f(loc + step_size * des_dir)
        g_hi = np.dot(des_dir, g(loc + step_size * des_dir))
        wolfe_success = check_wolfe(pdg_curr, g_hi, wolfe)
        wolfe_success = wolfe_success and \
                check_armijo(f_curr, f_hi, pdg_curr, armijo)

    # bracket refinement loop
    iter_num = 0

    # if we are here, f_hi,lo, g_hi,lo are all assigned
    while not wolfe_success and iter_num < max_iter:

        iter_num += 1
        # refine interval via cubic interpolant
        # cast to tuple for jit __find_cubic_interpolant_root__
        bracket_tup = tuple(bracket)
        step_size = __find_cubic_interpolant_root__(bracket_tup, f_hi, g_hi,
                                                        f_lo, g_lo)

        # pick which side to truncate
        new_grad_val = g(loc + step_size * des_dir)
        pdg_new = np.dot(des_dir, new_grad_val)
        f_new = f(loc + step_size * des_dir)
        if pdg_new < wolfe * np.abs(pdg_curr):
            # this is a better lower bound on interval
            bracket[0] = step_size
            g_lo = pdg_new
            f_lo = f_new
        else:
            # better as upper bound
            bracket[1] = step_size
            g_hi = pdg_new
            f_hi = f_new

        # check for success conditions
        wolfe_success = check_wolfe(pdg_new, pdg_curr, wolfe)
        #print("wolfe success: ", wolfe_success)
        #print("armijo success: ", check_armijo(f_new, f_curr, pdg_curr, armijo))
        wolfe_success = wolfe_success and \
                    check_armijo(f_new, f_curr, pdg_curr, armijo)

    # end of while loop, have we succeeded
    if not wolfe_success:
        print("Warning: wolfe search could not satisfy both ")
        print("wolfe and armijo conditions")

    # spit out the appropriate step size
    # if wolfe_success, this is the intermediate value determined above
    return step_size
