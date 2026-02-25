import cvxpy as cp
import numpy as np
import time
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class CVXPY:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = get_default_g(self.alpha)
            self.f = get_default_f(self.K, self.target)
            self.j = lambda u: self.f(u) + self.g(u)

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        u = cp.Variable(self.K.shape[1])
        u.value = u_0
        obj = cp.Minimize(
            0.5 * cp.sum_squares(self.K @ u - self.target) + self.alpha * cp.norm(u, 1)
        )
        constraints = []
        problem = cp.Problem(obj, constraints)
        problem.solve(solver=cp.SCS, eps=tol)
        logging.debug(
            f"CVXPY in {self.K.shape[1]} dimensions converged to tolerance {tol:.3E}"
        )
        return u.value

    def solve_experiment(self, tol: float):
        time_0 = time.time()
        u = np.zeros(self.K.shape[1])
        times = [time.time() - time_0]
        objectives = [self.j(u)]
        k = 0
        while 10**k >= tol - self.machine_precision:
            u = self.solve(tol=10**k, u_0=u)
            k -= 1
            times.append(time.time() - time_0)
            objectives.append(self.j(u))
        return u, objectives, times
