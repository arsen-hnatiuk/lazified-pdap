# An implementation of the semismooth Newton method following https://epubs.siam.org/doi/epdf/10.1137/120892167

import numpy as np
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN:
    def __init__(
        self, K: np.ndarray, alpha: float, y_true: np.ndarray, M: float
    ) -> None:
        self.K = K
        self.y_true = y_true
        self.alpha = alpha
        self.beta = 0.1
        self.tau = 4
        self.gamma_filter = 0.07
        self.g = get_default_g(self.alpha)
        self.f = get_default_f(self.K, self.y_true)
        self.p = get_default_p(self.K, self.y_true)
        self.gradient_p = np.matmul(np.transpose(self.K), self.K)
        self.j = lambda u: self.f(u) + self.g(u)
        self.L = 1
        self.norm_K_star = max(
            [np.linalg.norm(row) for row in np.transpose(self.K)]
        )  # the 2,inf norm of the transpose of K
        self.gamma = 1
        self.M = M

    def projector(self, u: np.ndarray, factor: float = 1) -> np.ndarray:
        # Projecion of an array onto [-alpha, alpha]
        return np.minimum(factor * self.alpha, np.maximum(-factor * self.alpha, u))

    def F(self, u: np.ndarray) -> np.ndarray:
        g = -self.p(u)
        return g - self.projector(u=g - u / self.tau)

    def theta(self, u: np.ndarray) -> np.ndarray:
        return np.absolute(self.F(u))

    def G(self, u: np.ndarray) -> np.ndarray:
        return u + self.tau * self.p(u)

    def S(self, u: np.ndarray) -> np.ndarray:
        return u - self.projector(u, self.tau)

    # def M(self, u: np.ndarray) -> np.ndarray:
    #     # (I-D(u))H(u)+D(u)
    #     pre_D = np.absolute(self.p(u) - u) <= self.alpha
    #     D = np.diag(pre_D)
    #     return np.matmul(np.eye(len(pre_D)) - D, self.gradient_p) + D

    def Psi(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        p = self.p(u)
        constant_part = np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def delta(self, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        # -p(u)^T d + alpha(||S(G(u))||_1-||u||_1)
        return np.matmul(-self.p(u), d) + self.alpha * (
            np.linalg.norm(self.S(self.G(u)), ord=1) - np.linalg.norm(u, ord=1)
        )

    def armijo(self, u: np.ndarray, d: np.ndarray) -> float:
        # Armijo step size
        delta = self.delta(u, d)
        step_size = 1
        while self.j(u + step_size * d) - self.j(u) > step_size * 0.1 * delta:
            step_size *= self.beta
        return step_size

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized)
        u = u_0
        initial_j = self.j(u)
        filter = [self.theta(u)]

        counter = 0
        while self.Psi(u) > tol or self.j(u) > initial_j:
            counter += 1
            if counter == 2:
                break
            logging.debug(f"Psi: {self.Psi(u)}")

            # Semismooth Newton step
            condition = np.absolute(-self.p(u) - u) > self.alpha
            cal_A = np.where(condition)[0]
            cal_I = np.where(~condition)[0]
            F_value = self.F(u)
            s_I = -self.tau * F_value[cal_I]
            s_A = np.linalg.lstsq(
                self.gradient_p[np.ix_(cal_A, cal_A)]
                + np.linalg.norm(F_value) * np.eye(len(cal_A)),
                -F_value[cal_A] - np.matmul(self.gradient_p[np.ix_(cal_A, cal_I)], s_I),
                rcond=None,
            )[0]
            s = np.zeros(len(u))
            s[cal_A] = s_A
            s[cal_I] = s_I
            logging.debug(f"cal_A: {cal_A}")
            logging.debug(f"cal_I: {cal_I}")
            logging.debug(f"condition: {condition}")
            logging.debug(f"SSN: {s}")
            logging.debug(f"SSN step: {u + s}")

            # Check if filter accepts SSN step
            u_ssn = u + s
            accepted = True
            logging.debug(f"filter: {filter}")
            for q in filter:
                if np.max(q - self.theta(u_ssn)) < self.gamma_filter * np.max(
                    self.theta(u_ssn)
                ):
                    accepted = False
                    break
            if accepted:
                filter.append(self.theta(u_ssn))
                u = u_ssn
                continue

            # Global Armijo step
            d = self.S(self.G(u))
            step_size = self.armijo(u, d)
            u = u + step_size * d
            logging.debug(f"armijo step size: {step_size}")
            logging.debug(f"armijo step: {u}")
        return u
