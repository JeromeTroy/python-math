"""
Line search methods for optimization
"""

def execute_line_search(fun, grad, loc, dec_dir, default_step_size,
        armijo, wolfe):
    pass

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

def backtracking(fun, grad, loc, dec_dir, default_step_size, armijo,
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
        dec_dir : array of floats
            decent direction computed
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

    fun_old = f(loc)

    # pre do the dot product
    pdg = np.dot(grad, dec_dir)
    is_suff_decent = False

    while not is_suff_decent and ls_iters < max_iters:
        # update step size
        new_step_size /= 2

        # check armijo
        fun_new = f(loc + new_step_size * dec_dir)
        is_suff_decent = check_armijo(fun_new, fun_old, pdg, armijo)

        ls_iters += 1

    if ls_iters == max_iters:
        print("Warning, line search could not satisfy Armijo Condition")

    return new_step_size

def bracketing(grad, loc, dec_dir, default_step_size, max_iter, wolfe):
    """
    Bracketing to determine interval of valid solution

    Input:

    Output:
        bracket : [a, b], both floats
            bracket found so that optimal step size
            is in [a, b]
    """

    # setup bracket
    bracket = [default_step_size, 2 * default_step_size]
    bracket_success = False
    key_val = wolfe * np.abs(np.dot(dec_dir, grad(loc)))

    # gradients at each end
    lower_grad = np.dot(dec_dir, grad(bracket[0]))
    upper_grad = np.dot(dec_dir, grad(bracket[1]))

    iter_num = 0

    while not bracket_success and iter_num < max_iter:

        # what can be wrong
        if key_val >= -lower_grad:
            # lower gradient too steep
            if key_val < -upper_grad:
                # upper gradient can replace
                bracket[0] = bracket[1]

                # update values
                lower_grad = upper_grad
                bracekt[1] *= 2

                continue

            else:
                # upper gradient cannot replace
                # default to no step size
                bracket[0] = 0
                lower_grad = np.dot(dec_dir, grad(loc))

        if key_val >= upper_grad:
            # upper gradient is too shallow
            # go further
            bracket[1] *= 2
            upper_grad = np.dot(dec_dir, grad(bracket[1]))

        bracket_success = ((key_val < -lower_grad) and
                (key_val < upper_grad))
        iter_num += 1

    if not bracket_success:
        print("Warning wolfe condition cannot be satisfied")

    return bracket



    return bracket
