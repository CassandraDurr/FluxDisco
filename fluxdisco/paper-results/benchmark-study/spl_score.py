"""Base scoring code from the Symbolic Physics Learner repository."""

# ------------------------------------------------------------------------------------
# Code from https://github.com/isds-neu/SymbolicPhysicsLearner/blob/main/score.py
# No option to pip install the package, so we copy the relevant code here.
# ------------------------------------------------------------------------------------

import _thread
import threading
from contextlib import contextmanager

import numpy as np
from numpy import *
from scipy.optimize import minimize
from sympy import expand, simplify


class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=""):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    """
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    """
    eq = simplify_eq(eq)
    if "C**" in eq:
        c_poly = ["C**" + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, "C")
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999):
    ## define independent variables and dependent variable
    num_var = data.shape[0] - 1

    if num_var <= 3:  ## most cases ([x], [x,y], or [x,y,z])
        current_var = "x"
        for i in range(num_var):
            globals()[current_var] = data[i, :]
            current_var = chr(ord(current_var) + 1)
        globals()["f_true"] = data[-1, :]
        f_true = data[-1, :]  # <-- CHANGE: Explicit local assignment
    else:  ## currently only double pendulum case has more than 3 independent variables
        print("WARNING: More than 3 variables detected.")
        globals()["x1"] = data[0, :]
        globals()["x2"] = data[1, :]
        globals()["w1"] = data[2, :]
        globals()["w2"] = data[3, :]
        globals()["wdot"] = data[4, :]
        globals()["f_true"] = data[5, :]
        f_true = data[5, :]  # <-- CHANGE: Explicit local assignment

    ## count number of numerical values in eq
    c_count = eq.count("C")
    with time_limit(t_limit, "sleep"):
        try:
            if c_count == 0:  ## no numerical values
                f_pred = eval(eq)
            elif c_count >= 10:  ## discourage over complicated numerical estimations
                return 0, eq
            else:  ## with numerical values: coefficient estimation with Powell method
                c_lst = ["c" + str(i) for i in range(c_count)]
                for c in c_lst:
                    eq = eq.replace("C", c, 1)

                def eq_test(c):
                    for i in range(len(c)):
                        globals()["c" + str(i)] = c[i]
                    # f_true is now safely accessed from the outer local scope
                    return np.linalg.norm(eval(eq) - f_true, 2)

                x0 = [1.0] * len(c_lst)
                c_lst = minimize(eq_test, x0, method="Powell", tol=1e-6).x.tolist()
                c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                eq_est = eq
                for i in range(len(c_lst)):
                    eq_est = eq_est.replace("c" + str(i), str(c_lst[i]), 1)
                eq = eq_est.replace("+-", "-")
                f_pred = eval(eq)
        except:
            return 0, eq

    # f_true is safely accessed here as well
    r = float(
        eta**tree_size
        / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0])
    )

    return r, eq
