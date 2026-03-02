import numpy as np
import logging
import time
from lib.default_values import *
from lib.ssn import SSN
from lib.sklearn_solver import SKLEARN
from lib.measure import Measure

logging.basicConfig(
    level=logging.DEBUG,
)


class LazifiedPDAPFinite:
    # An implementation of the LGCG algorithm for finite Omega

    def __init__(
        self,
        target: np.ndarray,
        K_transpose: np.ndarray,
        alpha: float = 1,
    ) -> None:
        self.K_transpose = K_transpose
        self.target = target
        self.f = (
            lambda u: 0.5
            * np.linalg.norm(u.duality_pairing(self.K_transpose) - target) ** 2
        )
        self.residuum = lambda u: u.duality_pairing(self.K_transpose) - self.target
        self.alpha = alpha
        self.g = lambda u: alpha * np.linalg.norm(u.coefficients, ord=1)
        self.L = 1
        self.norm_K = np.max(np.linalg.norm(self.K_transpose, axis=1))
        self.u_0 = Measure()
        self.j = lambda u: self.f(u) + self.g(u)
        self.M = self.j(self.u_0) / self.alpha  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2 * self.norm_K**2  # Smoothness constant
        self.machine_precision = 1e-12

    def Phi(self, p_u: np.ndarray, u: Measure, x: int) -> float:
        # M*max{0,||p_u||-alpha}+g(u)-<p_u,u>
        return (
            self.M * (max(0, np.abs(p_u[x]) - self.alpha))
            + self.g(u)
            - u.duality_pairing(p_u)
        )

    def finite_dimensional_step(self, u_plus: Measure, Psi: float) -> Measure:
        if not len(u_plus.coefficients):
            return u_plus
        K_support = self.K_transpose[u_plus.support.flatten()].T
        ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
        u_raw = ssn.solve(tol=Psi, u_0=u_plus.coefficients)
        if ssn.Psi(u_raw) > Psi:
            sklearn = SKLEARN(K=K_support, alpha=self.alpha, target=self.target)
            u_raw = sklearn.solve(tol=self.machine_precision)[0]
        u_raw[np.abs(u_raw) < self.machine_precision] = 0
        # Reconstruct u
        u = Measure(
            support=u_plus.support[u_raw != 0].copy(),
            coefficients=u_raw[u_raw != 0].copy(),
        )
        return u

    def solve(self, tol: float) -> dict:
        u = self.u_0 * 1
        residuum_u = self.residuum(u)
        p_u = -self.K_transpose @ residuum_u
        x = np.argmax(np.abs(p_u))
        epsilon = 0.5 * self.j(u) / self.M
        Psi = self.M * epsilon  # epsilon
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        objectives = [self.j(u)]
        times = [time.time() - start_time]
        while Phi_value > tol:
            Psi = max(min(0.5 * Phi_value, Psi), 2 * self.machine_precision)
            if abs(p_u[x]) < self.alpha:
                v_k = Measure()
            else:
                v_k = Measure(support=[[x]], coefficients=[self.M * np.sign(p_u[x])])
            eta = max(min(1, Phi_value / self.C), 10 * self.machine_precision)
            u = u * (1 - eta) + v_k * eta

            u = self.finite_dimensional_step(u, Psi)
            residuum_u = self.residuum(u)
            p_u = -self.K_transpose @ residuum_u
            x = np.argmax(np.abs(p_u))
            Phi_value = self.Phi(p_u, u, x)
            self.M = self.j(u) / self.alpha

            # logging.info(
            #     f"{k}: Phi {Phi_value:.3E}, epsilon {epsilon:.3E}, support {u.support}, Psi {Psi:.3E}, x: {x}"
            # )
            objectives.append(self.j(u))
            times.append(time.time() - start_time)
            k += 1
        logging.info(
            f"Finite LPDAP converged in {k} iterations and {time.time()-start_time:.3f}s to tolerance {tol:.3E} with final sparsity of {len(u.support)} and objective {objectives[-1]:.12E}"
        )
        return u, objectives, times

    def solve_exact(self, tol: float) -> dict:
        u = self.u_0 * 1
        residuum_u = self.residuum(u)
        p_u = -self.K_transpose @ residuum_u
        x = np.argmax(np.abs(p_u))
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        objectives = [self.j(u)]
        times = [time.time() - start_time]
        while Phi_value > tol:
            eta = max(min(1, Phi_value / self.C), 10 * self.machine_precision)
            v_k = Measure(support=[[x]], coefficients=[self.M * np.sign(p_u[x])])
            u_plus = u * (1 - eta) + v_k * eta
            u = self.finite_dimensional_step(u_plus, self.machine_precision)
            residuum_u = self.residuum(u)
            p_u = -self.K_transpose @ residuum_u
            x = np.argmax(np.abs(p_u))
            Phi_value = self.Phi(p_u, u, x)
            self.M = self.j(u) / self.alpha

            # logging.info(f"{k}: Phi {Phi_value:.3E}, support {len(u.support)}")
            objectives.append(self.j(u))
            times.append(time.time() - start_time)
            k += 1
        logging.info(
            f"Finite PDAP converged in {k} iterations and {time.time()-start_time:.3f}s to tolerance {tol:.3E} with final sparsity of {len(u.support)} and objective {objectives[-1]:.12E}"
        )
        return u, objectives, times
