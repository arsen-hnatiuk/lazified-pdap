import numpy as np
import time
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class FISTA:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.target = target
            self.alpha = alpha
            self.g = get_default_g(self.alpha)
            self.f = get_default_f(self.K, self.target)
            self.negative_grad_f = get_default_p(self.K, self.target)
            self.hessian = get_default_hessian(self.K)
            self.L = np.max(np.abs(np.linalg.eigvals(self.hessian)))
            self.j = lambda u: self.f(u) + self.g(u)

    def prox(self, q: np.ndarray) -> np.ndarray:
        inner_value = q + self.negative_grad_f(q) / self.L
        to_return = np.zeros(q.shape)
        for i, val in enumerate(inner_value):
            if np.abs(val) > self.alpha / self.L:
                to_return[i] = val - self.alpha * np.sign(val) / self.L
        return to_return

    def solve(self, max_iter: int = 1000):
        time_0 = time.time()
        u = np.zeros(self.K.shape[1])
        q = u.copy()  # intermediate iterate
        t = 1
        times = [time.time() - time_0]
        objectives = [self.j(u)]
        for k in range(max_iter):
            u_plus = self.prox(q)
            t_plus = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            q = u + ((t - 1) / t_plus) * (u_plus - u)
            t = t_plus
            u = u_plus
            obj = self.j(u)
            times.append(time.time() - time_0)
            objectives.append(obj)
            if k % 1000 == 0:
                logging.info(f"{k}: objective: {obj:.14E}")
        return u, objectives, times
