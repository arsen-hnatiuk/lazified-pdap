import numpy as np
import time
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
        M: float,
        mode: str = "unconstrained",
        # mode: "unconstrained" for unconstrained, else for positive solutions
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = get_default_g(self.alpha)
            self.f = get_default_f(self.K, self.target)
            self.p = get_default_p(self.K, self.target)  # -f'
            self.hessian = get_default_hessian(self.K)
            self.j = lambda u: self.f(u) + self.g(u)
            self.M = M
            self.maximum_iterations = 1000
            if mode == "unconstrained":
                self.Psi = self.Psi_unconstrained
                self.prox = self.prox_unconstrained
                self.grad_prox = self.grad_prox_unconstrained
            else:
                self.Psi = self.Psi_positive
                self.prox = self.prox_positive
                self.grad_prox = self.grad_prox_positive

    def Psi_unconstrained(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def Psi_positive(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(p) - self.alpha))
        return constant_part + variable_part

    def prox_unconstrained(self, q: np.ndarray, c: float = 1) -> np.ndarray:
        return np.sign(q) * np.maximum(np.abs(q) - self.alpha / c, 0)

    def prox_positive(self, q: np.ndarray, c: float = 1) -> np.ndarray:
        return np.sign(q) * np.maximum(q - self.alpha / c, 0)

    def grad_prox_unconstrained(self, q: np.ndarray, c: float = 1) -> np.ndarray:
        return np.diag(np.where(np.abs(q) > self.alpha / c, 1, 0))

    def grad_prox_positive(self, q: np.ndarray, c: float = 1) -> np.ndarray:
        return np.diag(np.where(q > self.alpha / c, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray, do_logging: bool = True) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0  # + self.p(u_0)
        prox_q = self.prox(q)  # The actual iterate
        psi_val = min(self.Psi(prox_q), self.Psi(q))
        k = 0
        while psi_val > tol:
            if k > self.maximum_iterations:
                logging.warning(
                    f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {psi_val:.3E} achieved, theta: {theta}, qdiff: {qdiff}"
                )
                if self.j(prox_q) <= initial_j:
                    return prox_q
                else:
                    return u_0
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian - Id) @ self.grad_prox(q)
            theta = theta / 10
            qdiff = tol + 1
            while qdiff >= self.machine_precision:
                theta = 2 * theta
                try:
                    direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                except np.linalg.LinAlgError:
                    logging.warning(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {psi_val:.3E} achieved, theta: {theta}, qdiff: {qdiff}"
                    )
                    if self.j(prox_q) <= initial_j:
                        return prox_q
                    else:
                        return u_0
                qnew = q - direction
                prox_qnew = self.prox(qnew)
                qdiff = self.j(prox_qnew) - self.j(prox_q)
            q = qnew
            prox_q = prox_qnew
            psi_val = self.Psi(prox_q)
            k += 1

        if do_logging:
            logging.info(
                f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
            )
        if k == 0:
            if self.j(q) < self.j(prox_q):
                return q
        return prox_q

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
