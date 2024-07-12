# An implementation of the semismooth Newton method following https://epubs.siam.org/doi/epdf/10.1137/120892167
# And Section 3 of https://mediatum.ub.tum.de/doc/1241413/1241413.pdf

import numpy as np
import logging
from lib.default_values import *

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN:
    def __init__(
        self, K: np.ndarray, alpha: float, target: np.ndarray, M: float
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

    def Psi(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def prox(self, q: np.ndarray, alpha: float) -> np.ndarray:
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if np.abs(val) > alpha:
                to_return[i] = val - alpha * np.sign(val)
        return to_return

    def grad_prox(self, q: np.ndarray, alpha: float) -> np.ndarray:
        return np.diag(np.where(np.abs(q) > alpha, 1, 0))

    def rebalance(self, tol: float, current_u: np.ndarray) -> np.ndarray:
        # Algorithm makes no progress, probably singular hessian
        # Remove columns to assure better stability
        logging.debug("Rebalancing columns")
        size = self.K.shape[1]
        K_candidate = self.K.copy()
        value_candidate = min(np.linalg.eigvals(K_candidate.T @ K_candidate))
        index_candidate = -1
        for i in range(size):
            indices = [j for j in range(size) if j != i]
            new_K = self.K.T[indices].T
            value = min(np.linalg.eigvals(new_K.T @ new_K))
            if value > value_candidate:
                K_candidate = new_K
                value_candidate = value
                index_candidate = i
        if index_candidate == -1:
            logging.warning("SSN failed to converge")
            return
        new_ssn = SSN(K_candidate, self.alpha, self.target, self.M)
        new_solution = new_ssn.solve(
            tol, current_u[[j for j in range(size) if j != index_candidate]]
        )
        new_solution_left = new_solution[:index_candidate]
        new_solution_right = new_solution[index_candidate:]
        adjusted_solution = (
            new_solution_left.tolist() + [0] + new_solution_right.tolist()
        )
        return np.array(adjusted_solution)

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0 + self.p(u_0)
        prox_q = self.prox(q, self.alpha)  # The actual iterate
        k = 0
        while self.Psi(prox_q) > tol or self.j(prox_q) > initial_j:
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian - Id) @ self.grad_prox(q, self.alpha)
            theta = theta / 10
            direction = np.linalg.solve(left_hand + theta * Id, right_hand)
            qnew = q - direction
            prox_qnew = self.prox(qnew, self.alpha)

            # Backtracking line search
            qdiff = self.j(prox_qnew) - self.j(prox_q)
            while qdiff >= tol:
                theta = 2 * theta
                direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                qnew = q - direction
                prox_qnew = self.prox(qnew, self.alpha)
                qdiff = self.j(prox_qnew) - self.j(prox_q)

            q = qnew
            prox_q = prox_qnew
            k += 1
            if k > 1000:
                return self.rebalance(tol, prox_q)

        logging.debug(
            f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
        )
        return prox_q


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     u = np.array([-1, -1, -1])
#     y = np.array([1, 0, 4])
#     sn = SSN(K, 1, y, 20)
#     print(sn.solve(1e-12, u))
