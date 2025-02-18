# Copyright 2025 Gian Gentinetta - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import double, finfo, array, zeros, dot, size, inf, all, max, min, abs, tile
from numpy import linspace, triu, sqrt, allclose, sign
from numpy.linalg import norm
from scipy.optimize import fsolve

"""
This file contains different methods to numerically calculate the solutions of ordinary
differential equations of the form

	y'(t) = f(t,y)
	y(t0) = y0

using Runge-Kutta's Butcher-tableaus.
An overview:

	1) A List of useful Butcher-tableaus
	
	2) Functions for the explicit and implicit RK steps
	
	3) An integrator to solve the ode with a specific Butcher-tableau
	
	4) A general adaptive RK solver that takes two Butcher-tableaus of different order 
	   as input.
	
	5) A faster adaptive RK solver that takes a Butcher-tableau with an additional b vector
	   as input.

	6) Two examples of the above methods:

	   a) ode45 (Dormand-Prince method, explicit)
	   b) gauss46 (implicit solver using Gauss-Legendre tableaus)
	   

--
Gian Gentinetta, 2019
This file was created as part of my exercise classes in 'Numerical methods for physicists'
at ETH Zurich taught by Dr. Vasile Gradinaru.
"""

# A List of useful Butcher-Tableaus
Euler = array([[0, 0], [0, 1]])
Midpoint = array([[0, 0, 0], [1 / 2, 1 / 2, 0], [0, 0, 1]])

iEuler = array([[1, 1], [0, 1]])
iMidpoint = array([[1 / 2, 1 / 2], [0, 1]])


# Runge-Kutta-Fehlberg's 4(5) Butcher-tableau
RKF5 = array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 4, 1 / 4, 0, 0, 0, 0, 0],
        [3 / 8, 3 / 32, 9 / 32, 0, 0, 0, 0],
        [12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [1, 439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        [0, 16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
    ]
)

RKF4 = array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 4, 1 / 4, 0, 0, 0, 0, 0],
        [3 / 8, 3 / 32, 9 / 32, 0, 0, 0, 0],
        [12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
        [1, 439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
        [1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        [0, 25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0],
    ]
)

RKF_b_low = array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])

# Dormand-Prince

RKDP = array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 10, 3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [4 / 5, 44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [8 / 9, 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [1, 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [1, 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        [0, 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ]
)

RKDP_b_low = array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)


# Classic Runge-Kutta (RK4)
RK4 = array(
    [
        [0, 0, 0, 0, 0],
        [1 / 2, 1 / 2, 0, 0, 0],
        [1 / 2, 0, 1 / 2, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1 / 6, 2 / 6, 2 / 6, 1 / 6],
    ]
)


# Kutta-3/8 rule
Kutta38 = array(
    [
        [0, 0, 0, 0, 0],
        [1 / 3, 1 / 3, 0, 0, 0],
        [2 / 3, -1 / 3, 1, 0, 0],
        [1, 1, -1, 1, 0],
        [0, 1 / 8, 3 / 8, 3 / 8, 1 / 8],
    ]
)

# Gauss-Legendre methods (implicit)
Gauss4 = array(
    [
        [1 / 2 - sqrt(3) / 6, 1 / 4, 1 / 4 - sqrt(3) / 6],
        [1 / 2 + sqrt(3) / 6, 1 / 4 + sqrt(3) / 6, 1 / 4],
        [0, 1 / 2, 1 / 2],
    ]
)

Gauss4_b_low = array([1 / 2 + 1 / 2 * sqrt(3), 1 / 2 - 1 / 2 * sqrt(3)])


Gauss6 = array(
    [
        [1 / 2 - sqrt(15) / 10, 5 / 36, 2 / 9 - sqrt(15) / 15, 5 / 36 - sqrt(15) / 30],
        [1 / 2, 5 / 36 + sqrt(15) / 24, 2 / 9, 5 / 36 - sqrt(15) / 24],
        [1 / 2 + sqrt(15) / 10, 5 / 36 + sqrt(15) / 30, 2 / 9 + sqrt(15) / 15, 5 / 36],
        [0, 5 / 18, 4 / 9, 5 / 18],
    ]
)


# -------------------------------------------------------------------------------------- #
# Runge-Kutta solvers
def nicer_fsolve(F, initial_guess):
    """Wrapper for `scipy.optimize.fsolve`.

    This wrapper is used to solve non-linear equations of multidimensial arrays.
    """
    shape = initial_guess.shape

    initial_guess = initial_guess.reshape((-1,))
    result = fsolve(lambda x: F(x.reshape(shape)).reshape((-1)), initial_guess)

    return result.reshape(shape)


def nicer_norm(y):
    """
    Takes the infinity norm for an array y, where y does not have to be 1D or 2D.
    """
    return norm(y.reshape(-1), inf)


def exp_RKstep(B, f, y0, t0, h, ret_k=False):
    """
    Calculates y(t0 + h) using a given an explicit Butcher-tableau B and its corresponding
    Runge-Kutta method.

    --------------------------------------------------------------------------------------
    Input:

    B:  Butcher-tableau. For knots c, weights b and the matrix A this should look like

        B = c | A
                ------
                0 | b^T

        , where Aij = 0 for i <= j (explicit method).

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Starting vector at time t = t0

    t0: Initial time

    h:  Step size

    ret_k: Set to True to return k

    --------------------------------------------------------------------------------------
    Returns:

    y: The result y(t0 + h)
    (k: Function calls)
    """
    y0 = array(y0)

    shape = y0.shape  # Reshaping needs to be done in case y0 is 2D or larger
    y0 = y0.reshape(-1)  # as np.dot would misbehave otherwise.

    c = B[:-1, 0]  # Knots
    b = B[-1, 1:]  # Weights
    A = B[:-1, 1:]  # Matrix A

    s = len(c)  # Number of Knots
    d = len(y0)  # Dimension of y

    k = zeros((s, d))
    for i in range(s):
        k[i, :] = f(t0 + h * c[i], (y0 + h * dot(A[i, :], k)).reshape(shape)).reshape(
            -1
        )
        # Here we shape back so that rhs receives the original shape.

    y = y0 + h * dot(b, k)

    # Return k if specified in the argument.
    if ret_k:
        return y.reshape(shape), k

    return y.reshape(shape)


def imp_RKstep(B, f, y0, t0, h, ret_k=False, guess=None):
    """
    Calculates y(t0 + h) using a given an implicit Butcher-tableau B and its corresponding
    Runge-Kutta method.

    --------------------------------------------------------------------------------------
    Input:

    B:  Butcher-tableau. For knots c, weights b and the matrix A this should look like

        B = c | A
                ------
                0 | b^T

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Starting vector at time t = t0

    t0: Initial time

    h:  Step size

    --------------------------------------------------------------------------------------
    Returns:

    y: The result y(t0 + h)
    """
    y0 = array(y0)

    shape = y0.shape
    y0 = y0.reshape(-1)

    c = B[:-1, 0]  # Knots
    b = B[-1, 1:]  # Weights
    A = B[:-1, 1:]  # Matrix A

    s = len(c)  # Number of Knots
    d = len(y0)  # Dimension of y

    def F(x):
        k_x = zeros((s, d))
        for i in range(s):
            k_x[i, :] = f(
                t0 + h * c[i], (y0 + h * dot(A[i, :], x)).reshape(shape)
            ).reshape(-1)
        return k_x

    if guess is None:
        guess = zeros((s, d))
    else:
        guess = tile(guess, (s, 1))
    k = nicer_fsolve(
        lambda x: F(x) - x, guess
    )  # Solving the non-linear system of equations
    y = y0 + h * dot(b, k)

    # Return k if specified in the argument.
    if ret_k:
        return y.reshape(shape), k

    return y.reshape(shape)


def rk(B, f, tspan, y0, N):
    """
    Calculates the solution of an ordinary differential
    equation with right-hand-side f

            y'(t) = f(t,y)
            y(t0) = y0

    using a the Runge-Kutta method encoded in the Butcher-tableau B
    --------------------------------------------------------------------------------------
    Input:

    B: Butcher-tableau of the two Runge-Kutta method used to calculate the
       solution. For knots c, weights b and the matrix A this should look like

       B = c | A
               ------
               0 | b^T

       , where Aij = 0 for i <= j (explicit method).


    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Initial value y(t0) = y0

    tspan: Tuple (t_0,t_end) of start and end times


    --------------------------------------------------------------------------------------
    Returns: t, y

    t: The times at which y(t) is calculated
    y: The calculated solutions y(t)

    """

    y0 = array(y0, ndmin=1)

    if allclose(
        triu(B[:-1, 1:]), zeros(B[:-1, 1:].shape)
    ):  # Test if the method is explicit
        method = exp_RKstep
    else:
        method = imp_RKstep

    # Initialization
    t0, tend = tspan
    t, h = linspace(t0, tend, N + 1, retstep=True)
    y = zeros((N + 1,) + y0.shape)
    y[0, ...] = y0

    # Integrating the solution
    for i in range(N):
        y[i + 1, ...] = method(B, f, y[i, ...], t[i], h)

    return t, y


# -------------------------------------------------------------------------------------- #
# Adaptive Runge-Kutta solvers


def rk_adapt(
    B_high,
    B_low,
    f,
    tspan,
    y0,
    reltol=1e-6,
    abstol=1e-6,
    initialstep=None,
    maxstep=None,
):
    """
    Adaptive Runge-Kutta method. Calculates the solution of an ordinary differential
    equation with right-hand-side f:

            y'(t) = f(t,y)
            y(t0) = y0

    --------------------------------------------------------------------------------------
    Input:

    B_high, B_low: Butcher-tableaus of the two Runge-Kutta methods used to calculate the
                               solution. For knots c, weights b and the matrix A this should look like

                       B = c | A
                           ------
                           0 | b^T

                   , where Aij = 0 for i <= j (explicit method).

                   Examples are B_high = RKF5, B_low = RKF4

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Initial value y(t0) = y0

    tspan: Tuple (t_0,t_end) of start and end times

    --------------------------------------------------------------------------------------

    Optional Input:

    reltol: Relative tolerance

    abstol: Absolute tolerance

    initialstep: Set to a float if a specific initial step size is wished

    maxstep: Set to a float if the adaptive method should not be larger than some maxstep

    --------------------------------------------------------------------------------------
    Returns: t, y

    t: The times at which y(t) is calculated
    y: The calculated solutions y(t)

    """
    # Check if methods are implicit or explicit:

    if allclose(triu(B_high[:-1, 1:]), zeros(B_high[:-1, 1:].shape)):
        method_high = exp_RKstep
    else:
        method_high = imp_RKstep
        print("implicit")

    if allclose(triu(B_low[:-1, 1:]), zeros(B_low[:-1, 1:].shape)):
        method_low = exp_RKstep
    else:
        method_low = imp_RKstep

    # Ensure tspan and y0 are vectors:

    tspan = array(tspan, ndmin=1)
    y0 = array(y0, ndmin=1)

    # Setting the initial step, if not specified in the arguments.
    if initialstep == None:
        initialstep = double(tspan[-1] - tspan[0]) / 100

    # Setting the maximal step, if not specified in the arguments.
    if maxstep == None:
        maxstep = double(tspan[-1] - tspan[0]) / 10

    # Intitialization of the adaptive solver

    t0 = tspan[0]  # start time
    tend = tspan[-1]  # end time

    dir = sign(tend)  # Determine if integrating forwards or backwards in time

    h = initialstep  # step size

    epsilon = finfo(double).eps  # The smallest possible double in Python

    hmin = epsilon * double(tend - t0)  # The smallest step size used in this algorithm

    t = [t0]  # This will be the output list for the times
    y = [y0]  # This will be the output list for y(t).

    p = 0.2  # This is the exponent used to increase/decrease the stepsize

    # This is the main loop. It will stop if t = tend has been reached,
    # or if the step size cannot be reduced any further.

    while dir * t0 < dir * tend and dir * h >= dir * hmin:

        # If t0 + h surpasses the endpoint, we want to hit tend instead
        if t0 + h > dir * tend:
            h = tend - dir * t0

        # Estimating y(t0 + h) with our two methods
        y_low = method_low(B_low, f, y0, t0, h)
        y_high = method_high(B_high, f, y0, t0, h)

        # Comparing the two results
        err = nicer_norm(y_high - y_low)

        # If the error is within the tolerance, we set our new values for t0 and y0
        # and append them to our return lists.
        tol = max([reltol * max([nicer_norm(y0), 1.0]), abstol])

        if err <= tol:
            t0 = t0 + h
            t.append(t0)
            y0 = y_high
            y.append(y0)

        # We are dividing by the error, so we have to ensure that it is not zero:
        if err == 0:
            err = tol * (
                0.4 ** (1.0 / p)
            )  # This doubles the step size for the next step

        # Now we update the step size for the next iteration.
        if dir == 1:
            h = min([maxstep, min(0.8 * h * (tol / err) ** p)])
        else:
            h = max([maxstep, max(0.8 * h * (tol / err) ** p)])

    return array(t), array(y)


# ------------------------------------------------------------------------------------------ #
# Adaptive Runge-Kutta Method, where only b differs in the two Butcher-Tableaus
def fast_rk_adapt(
    B_high,
    b_low,
    f,
    tspan,
    y0,
    reltol=1e-6,
    abstol=1e-6,
    initialstep=None,
    maxstep=None,
):
    """
    Adaptive Runge-Kutta method. Calculates the solution of an ordinary differential
    equation with right-hand-side f:

            y'(t) = f(t,y)
            y(t0) = y0

    --------------------------------------------------------------------------------------
    Input:

    B_high:        Butcher-tableau of the Runge-Kutta method used to calculate the
                               solution. For knots c, weights b and the matrix A this should look like

                       B = c | A
                           ------
                           0 | b^T

                   , where Aij = 0 for i <= j (explicit method).

    b_low: Additional b vector to provide a second (less accurate) approximation

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Initial value y(t0) = y0

    tspan: Tuple (t_0,t_end) of start and end times

    --------------------------------------------------------------------------------------

    Optional Input:

    reltol: Relative tolerance

    abstol: Absolute tolerance

    initialstep: Set to a float if a specific initial step size is wished

    maxstep: Set to a float if the adaptive method should not be larger than some maxstep

    --------------------------------------------------------------------------------------
    Returns: t, y

    t: The times at which y(t) is calculated
    y: The calculated solutions y(t)

    """
    # Check if methods are implicit or explicit:

    if allclose(triu(B_high[:-1, 1:]), zeros(B_high[:-1, 1:].shape)):
        method_high = exp_RKstep
    else:
        method_high = imp_RKstep
        print("implicit")

    # Ensure tspan and y0 are vectors:

    tspan = array(tspan, ndmin=1)
    y0 = array(y0, ndmin=1)

    # Setting the initial step, if not specified in the arguments.
    if initialstep == None:
        initialstep = double(tspan[-1] - tspan[0]) / 100

    # Setting the maximal step, if not specified in the arguments.
    if maxstep == None:
        maxstep = double(tspan[-1] - tspan[0]) / 10

    # Intitialization of the adaptive solver

    t0 = tspan[0]  # start time
    tend = tspan[-1]  # end time

    dir = sign(tend)  # Determine if integrating forwards or backwards in time

    h = initialstep  # step size

    epsilon = finfo(double).eps  # The smallest possible double in Python

    hmin = epsilon * double(tend - t0)  # The smallest step size used in this algorithm

    t = [t0]  # This will be the output list for the times
    y = [y0]  # This will be the output list for y(t).

    p = 0.2  # This is the exponent used to increase/decrease the stepsize

    # This is the main loop. It will stop if t = tend has been reached,
    # or if the step size cannot be reduced any further.

    while dir * t0 < dir * tend and dir * h >= dir * hmin:

        # If t0 + h surpasses the endpoint, we want to hit tend instead
        if t0 + h > dir * tend:
            h = tend - dir * t0

        # Estimating y(t0 + h) with our two methods
        y_high, k = method_high(B_high, f, y0, t0, h, ret_k=True)
        # Repeat last part of the RK-step for the second b vector
        y_low = y0 + h * dot(b_low, k).reshape(y0.shape)

        # Comparing the two results
        err = nicer_norm(y_high - y_low)

        # If the error is within the tolerance, we set our new values for t0 and y0
        # and append them to our return lists.
        tol = max([reltol * max([nicer_norm(y0), 1.0]), abstol])

        if err <= tol:
            t0 = t0 + h
            t.append(t0)
            y0 = y_high
            y.append(y0)

        # We are dividing by the error, so we have to ensure that it is not zero:
        if err == 0:
            err = tol * (
                0.4 ** (1.0 / p)
            )  # This doubles the step size for the next step

        # Now we update the step size for the next iteration.
        if dir == 1:
            h = min([maxstep, min(0.8 * h * (tol / err) ** p)])
        else:
            h = max([maxstep, max(0.8 * h * (tol / err) ** p)])

    return array(t), array(y)


# --------------------------------------------------------------------------------------- #
# Examples:


def ode45(f, tspan, y0, reltol=1e-6, abstol=1e-6, initialstep=None, maxstep=None):
    """
    Calculates the solution of an ordinary differential equation with right-hand-side f

            y'(t) = f(t,y)
            y(t0) = y0

    using Runge-Kutta-Fehlberg's 4(5) method.

    --------------------------------------------------------------------------------------
    Input:

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Initial value y(t0) = y0

    tspan: Tuple (t_0,t_end) of start and end times

    --------------------------------------------------------------------------------------

    Optional Input:

    reltol: Relative tolerance

    abstol: Absolute tolerance

    initialstep: Set to a float if a specific initial step size is wished

    maxstep: Set to a float if the adaptive method should not be larger than some maxstep

    --------------------------------------------------------------------------------------
    Returns: t, y

    t: The times at which y(t) is calculated
    y: The calculated solutions y(t)

    """

    return fast_rk_adapt(
        RKDP,
        RKDP_b_low,
        f,
        tspan,
        y0,
        reltol=reltol,
        abstol=abstol,
        initialstep=initialstep,
        maxstep=maxstep,
    )


def gauss46(f, tspan, y0, reltol=1e-6, abstol=1e-6, initialstep=None, maxstep=None):
    """
    Calculates the solution of an ordinary differential equation with right-hand-side f

            y'(t) = f(t,y)
            y(t0) = y0

    using an adaptive Runge-Kutta method with the Gauss-Legendre Butcher-tableaus of order
    6 and 4. As an implicit method this should be used to find solutions of stiff
    equations.

    --------------------------------------------------------------------------------------
    Input:

    f: Right-hand-side of the differential equation y'(t) = f(t,y)

    y0: Initial value y(t0) = y0

    tspan: Tuple (t_0,t_end) of start and end times

    --------------------------------------------------------------------------------------

    Optional Input:

    reltol: Relative tolerance

    abstol: Absolute tolerance

    initialstep: Set to a float if a specific initial step size is wished

    maxstep: Set to a float if the adaptive method should not be larger than some maxstep

    --------------------------------------------------------------------------------------
    Returns: t, y

    t: The times at which y(t) is calculated
    y: The calculated solutions y(t)

    """

    return rk_adapt(
        Gauss6,
        Gauss4,
        f,
        tspan,
        y0,
        reltol=reltol,
        abstol=abstol,
        initialstep=initialstep,
        maxstep=maxstep,
    )

    """
	Changelog:
	29.03.2020: Added reshapes to support more cases of y0 shapes. Added nicer_norm for similar purposes.
	31.03.2020: Added faster variant for the adaptive method, added butcher tableaus for Dormand-Prince.
	"""
