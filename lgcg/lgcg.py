import numpy as np
import logging
from typing import Callable
from lib.default_values import *
from sklearn.linear_model import Lasso

logging.basicConfig(
    level=logging.DEBUG,
)


class LGCG:
    def __init__(
        self,
        M: float,
        R: float,
        theta: float,
        y_true: np.ndarray = None,
        K: np.ndarray = None,
        f: Callable = None,
        p: Callable = None,
        alpha: float = None,
        gamma: float = None,
        L: float = None,
        norm_K_star: float = None,
        Omega: np.ndarray = None,
    ) -> None:
        self.M = M
        self.R = R
        self.theta = theta
        self.y_true = y_true if not y_true is None else np.random.rand(5)
        self.K = K if not K is None else get_default_K(self.y_true)
        self.f = f if not f is None else get_default_f(self.K, self.y_true)
        self.p = p if not p is None else get_default_p(self.K, self.y_true)
        self.alpha = alpha if not alpha is None else 1
        self.g = get_default_g(self.alpha)
        self.gamma = gamma if not gamma is None else 1
        self.L = L if not L is None else 1
        self.norm_K_star = (
            norm_K_star if not norm_K_star is None else np.linalg.norm(self.K, ord=1)
        )
        self.Omega = Omega if not Omega is None else get_default_Omega(self.K)
        self.C = self.L * self.M**2
        self.j = lambda u: self.f(u) + self.g(u)

    def update_epsilon(self, eta: float, epsilon: float) -> float:
        return (self.M * epsilon + 0.5 * self.C * eta**2) / (self.M + self.M * eta)

    def explicit_Phi(self, p: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        return np.matmul(p, v - u) + self.g(u) - self.g(v)

    def optimize(self, tol: float) -> dict:
        A = np.array([])
        u = np.zeros(self.K.shape[1])
        p_u = self.p(u)
        x = np.argmax(abs(p_u))
        max_p_u = abs(p_u[x])
        max_p_u_A = 0
        epsilon = self.j(u) / self.M
        k = 1
        while abs(max_p_u - max_p_u_A) > tol:
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            A_half = np.append(A, x).astype(int)
            A_half.sort()
            v = self.M * np.sign(p_u[x]) * np.eye(1, self.K.shape[1], x)[0]
            if self.explicit_Phi(p=p_u, u=u, v=v) >= self.M * epsilon:
                u = (1 - eta) * u + eta * v
            elif (
                self.explicit_Phi(p=p_u, u=u, v=np.zeros(self.K.shape[1]))
                >= self.M * epsilon
            ):
                u = (1 - eta) * u

            # P_A step
            lasso = Lasso(alpha=self.alpha / len(self.y_true))
            K_A = self.K[:, A_half]
            lasso.fit(K_A, self.y_true)
            u_raw = lasso.coef_

            A = A_half[u_raw != 0]
            u = np.zeros(len(u))
            for ind, pos in enumerate(A):
                u += u_raw[ind] * np.eye(1, len(u), pos)[0]
            p_u = self.p(u)
            x = np.argmax(abs(p_u))
            max_p_u = abs(p_u[x])
            if len(A):
                max_p_u_A = np.max(p_u[A])
            else:
                max_p_u_A = 0
                logging.warning("Empty response vector")
            k += 1
            logging.info(f"k:{k},  A:{A},  u_raw:{u_raw},  Ku:{np.matmul(self.K,u)}")
