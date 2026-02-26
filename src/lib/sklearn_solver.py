from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class SKLEARN:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
    ) -> None:
        self.K_raw = K
        if all(self.K_raw.shape):
            self.scaler = StandardScaler()
            self.K = self.K_raw  # self.scaler.fit_transform(self.K_raw)
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha / self.K.shape[0]
            self.g = get_default_g(alpha)
            self.f = get_default_f(self.K_raw, self.target)
            self.j = lambda u: self.f(u) + self.g(u)

    def solve(self) -> np.ndarray:
        t_0 = time.time()
        model = Lasso(
            alpha=self.alpha,
            fit_intercept=False,
            max_iter=10000,
        )
        model.fit(self.K, self.target)
        u = model.coef_
        obj = self.j(u)
        logging.debug(f"Scikit-learn in {self.K.shape[1]} dimensions converged")
        return u, obj, time.time() - t_0

    def solve_experiment(self, tol: float):
        time_0 = time.time()
        u = np.zeros(self.K.shape[1])
        times = [time.time() - time_0]
        objectives = [self.j(u)]
        k = 0
        while 10**k >= tol - self.machine_precision:
            model = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                tol=10**k,
                warm_start=True,
                max_iter=10000,
            )
            model.coef_ = u
            model.fit(self.K, self.target)
            u = model.coef_
            k -= 1
            times.append(time.time() - time_0)
            objectives.append(self.j(u))
            logging.debug(
                f"Scikit-learn in {self.K.shape[1]} dimensions converged to tolerance {10**k:.3E}"
            )
        return u, objectives, times
