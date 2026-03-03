import numpy as np
import scipy as sp
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
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = get_default_g(self.alpha)
            self.f = get_default_f(self.K, self.target)
            self.negative_grad_f = get_default_p(self.K, self.target)
            self.L = sp.linalg.svdvals(self.K)[0] ** 2
            self.j = lambda u: self.f(u) + self.g(u)

    def prox(self, q: np.ndarray) -> np.ndarray:
        inner_value = q + self.negative_grad_f(q) / self.L
        to_return = np.sign(inner_value) * np.maximum(
            np.abs(inner_value) - self.alpha / self.L, 0
        )
        return to_return

    def solve(self, max_time: int = 1000):
        time_0 = time.time()
        u = np.zeros(self.K.shape[1])
        q = u.copy()  # intermediate iterate
        t = 1
        times = [time.time() - time_0]
        objectives = [self.j(u)]
        k = 1
        while time.time() - time_0 < max_time:
            u_plus = self.prox(q)
            t_plus = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            q = u_plus + ((t - 1) / t_plus) * (u_plus - u)
            t = t_plus
            u = u_plus
            obj = self.j(u)
            times.append(time.time() - time_0)
            objectives.append(obj)
            # if k % 1000 == 0:
            #     logging.info(f"{k}: objective: {obj:.14E}")
            if np.std(objectives[-10:]) < 1e-12:
                break
            k += 1
        logging.info(f"FISTA reached {k} iterations with objective: {obj:.14E}")
        return u, objectives, times
