"""
finite difference methods for computing jacobians
(gradients and hessians)
"""

import numpy as np
import numba

@numba.jit(nopython=True)
def fdgrad(f, x, step=1e-6):
    """
    Finite difference approximation for gradient

    Input:
        f : callable, signature f(x)
            function to differentiate
        x : vector of floats
            point at which to compute grad f
        step : float, optional
            step size for gradient computation.
            The default is 1e-6
    Output:
        df : vector of floats
            grad[f](x)

    Warning!
    In order to make use of the jit feature of this function,
    the objective function f must also be decorated as
    numba.jit (just-in-time compiled)
    """

    n = x.shape[0]
    df = np.zeros((n))

    for index in range(n):
        x_up = np.copy(x)
        x_up[index] += step
        x_lo = np.copy(x)
        x_lo[index] -= step

        df[index] = (f(x_up) - f(x_lo)) / (2 * step)

    return df

@numba.jit(nopython=True)
def fdhess(f, x, step=1e-6):
    """
    Finite difference approximation for hessian from original function

    Input:
        f : callable, signature f(x)
            function to differentiate
        x : vector of floats
            point at which to compute hess f
        step : float, optional
            step size for gradient computation
            The default is 1e-6
    Output:
        Hf : array of floats
            H[f] (x)
    """

    n = x.shape[0]
    Hf = np.zeros((n, n))

    # start by assigning diagonal entries
    for index in range(n):
        x_up = np.copy(x)
        x_up[index] += step
        x_lo = np.copy(x)
        x_lo[index] -= step

        Hf[index, index] = (f(x_up) - 2 * f(x) + f(x_lo)) / (step ** 2)


    # off diagonals
    # Hf is symmetric, so only do for top half
    for row in range(n):
        for col in range(row+1, n):
            # df / dx[col] at (x + step e_row)
            xi_up = np.copy(x)
            xi_up[row] += step

            # derivative to 4'th order accuracy
            d4f_up = __pder_fourth_order__(f, xi_up, step, col)

            # df / dx[col] at (x - step e_row)
            xi_lo = np.copy(x)
            xi_lo[row] -= step

            # derivative to 4'th order accuracy
            d4f_lo = __pder_fourth_order__(f, xi_lo, step, col)

            # apply finite difference on previous derivatives
            der = (d4f_up - d4f_lo) / (2 * step)
            Hf[row, col] = der
            Hf[col, row] = der

    return Hf


@numba.jit(nopython=True)
def __pder_fourth_order__(f, x, step, index):
    """
    Fourth order accurate derivative computation

    Input:
        f : callable, signature f(x)
            function to differentiate
        x : vector of floats
            point at which to compute derivative
        step : float
            step size for finite difference
        index : int
            index for partial derivative df/dx[index]
    Output:
        df4dxi : float
            4'th order derivative computation

    Warning!
    In order to make use of the jit feature of this function,
    the objective function f must also be decorated as
    numba.jit (just-in-time compiled)
    """

    # evaluation locations
    x_2up = np.copy(x)
    x_2up[index] += 2 * step
    x_up = np.copy(x)
    x_up[index] += step
    x_lo = np.copy(x)
    x_lo[index] -= step
    x_2lo = np.copy(x)
    x_2lo[index] -= 2 * step

    # finite difference computation
    df4dxi = (-f(x_2up) + 8 * f(x_up) - 8 * f(x_lo) + f(x_2lo)) / (12 * step)

    return df4dxi

#@numba.jit(nopython=True)
def fdjacobian(f, x, step=1e-6):
    """
    Finite difference approximation for the jacobian

    Input:
        f : callable, signature f(x)
            vector function
            objective is || f(x) ||^2
        x : array of floats
            current position
        step: float
            finite difference step size
    Output:
        jac : matrix of floats
            jacobian of f
        jac[i, j] = df_i / dx_j (x)
    """

    # determine shape and build space
    n = x.shape[0]
    m = f(x).shape[0]
    jac = np.zeros((m, n))

    # iterate along input dim
    for input_index in range(n):
        x_up = np.copy(x)
        x_up[input_index] += step
        x_lo = np.copy(x)
        x_lo[input_index] -= step

        # assign column wise
        jac[:, input_index] = (f(x_up) - f(x_lo)) / (2 * step)

    return jac
