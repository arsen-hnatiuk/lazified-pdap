import numpy as np
import logging
import time
from lib.default_values import *
from lib.ssn import SSN
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
        K = np.append(
            K,
            np.ones((K.shape[0], 1)),
            axis=1,
        )  # Adding a constant term
        self.K_norms = np.linalg.norm(K_transpose, axis=1)
        self.K_norms[self.K_norms == 0] = 1  # Avoid division by zero
        self.K_transpose = np.divide(K_transpose, self.K_norms.reshape(-1, 1))
        self.target_norm = np.linalg.norm(target)
        self.target = target / self.target_norm
        self.f = (
            lambda u: 0.5
            * np.linalg.norm(u.duality_pairing(self.K_transpose) - target) ** 2
        )
        self.residuum = lambda u: u.duality_pairing(self.K_transpose) - self.target
        self.alpha = alpha
        self.g = lambda u: alpha * np.linalg.norm(u.coefficients, ord=1)
        self.L = 1
        self.norm_K = np.max(np.linalg.norm(self.K_transpose, axis=1))
        self.u_0 = Measure(
            support=[[self.K_transpose.shape[0] - 1]], coefficients=[1]
        )  # Only the constant term
        self.j = lambda u: self.f(u) + self.g(u)
        self.M = self.j(self.u_0) / self.alpha  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2 * self.norm_K**2  # Smoothness constant
        self.machine_precision = 1e-10

    def explicit_Phi(self, p: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        # <p(u),v-u>+g(u)-g(v)
        return (v - u).duality_pairing(p) + self.g(u) - self.g(v)

    def Phi(self, p_u: np.ndarray, u: Measure, x: int) -> float:
        # M*max{0,||p_u||-alpha}+g(u)-<p_u,u>
        return (
            self.M * (max(0, np.absolute(p_u[x]) - self.alpha))
            + self.g(u)
            - u.duality_pairing(p_u)
        )

    def finite_dimensional_step(self, u_plus: Measure, Psi: float) -> Measure:
        if not len(u_plus.coefficients):
            return u_plus
        K_support = self.K_transpose[u_plus.support.flatten()].T
        ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
        u_raw = ssn.solve(tol=Psi, u_0=u_plus.coefficients)
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
        x = np.argmax(np.absolute(p_u))
        epsilon = 0.5 * self.j(u) / self.M
        Psi = epsilon
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        ssn_time = 0
        while Phi_value > tol:
            u_old = u * 1
            Psi_old = Psi
            Psi = max(min(Psi, self.M * epsilon), self.machine_precision)
            if x in u.support:
                Psi = Psi / 2
            if abs(p_u[x]) < self.alpha:
                v_k = Measure()
            else:
                v_k = Measure(support=[[x]], coefficients=[self.M * np.sign(p_u[x])])
            Phi_x = self.explicit_Phi(p=p_u, u=u, v=v_k)
            eta = min(1, Phi_x / self.C)
            u = u * (1 - eta) + v_k * eta

            if not np.array_equal(u, u_old) or Psi_old != Psi:
                # Low-dimensional optimization
                ssn_start = time.time()
                u = self.finite_dimensional_step(u, Psi)
                ssn_time += time.time() - ssn_start

                if not np.array_equal(
                    u.coefficients, u_old.coefficients
                ) or not np.array_equal(u.support, u_old.support):
                    # SSN found a different solution
                    residuum_u = self.residuum(u)
                    p_u = -self.K_transpose @ residuum_u
                    x = np.argmax(np.absolute(p_u))
                    Phi_value = self.Phi(p_u, u, x)

            logging.info(
                f"{k}: Phi {Phi_value:.3E}, epsilon {epsilon:.3E}, support {u.support}, Psi {Psi:.3E}"
            )
            k += 1
        logging.info(
            f"LGCG converged in {k} iterations and {time.time()-start_time:.3f}s (SSN time {ssn_time:.3f}s) to tolerance {tol:.3E} with final sparsity of {len(u.support)}"
        )
        # Rescale the solution
        for ind, x in enumerate(u.support):
            u.coefficiens[ind] /= self.K_norms[x[0]]
        u = u * self.target_norm

        return u

    def solve_exact(self, tol: float) -> dict:
        u = self.u_0 * 1
        residuum_u = self.residuum(u)
        p_u = -self.K_transpose @ residuum_u
        x = np.argmax(np.absolute(p_u))
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        ssn_time = 0
        while Phi_value > tol:
            eta = min(1, Phi_value / self.C)
            v_k = Measure(support=[[x]], coefficients=[self.M * np.sign(p_u[x])])
            u_plus = u * (1 - eta) + v_k * eta
            ssn_start = time.time()
            u = self.finite_dimensional_step(u_plus, self.machine_precision)
            ssn_time += time.time() - ssn_start
            residuum_u = self.residuum(u)
            p_u = -self.K_transpose @ residuum_u
            x = np.argmax(np.absolute(p_u))
            Phi_value = self.Phi(p_u, u, x)

            logging.info(f"{k}: Phi {Phi_value:.3E}, support {u.support}")
            k += 1
        logging.info(
            f"LGCG converged in {k} iterations and {time.time()-start_time:.3f}s (SSN time {ssn_time:.3f}s) to tolerance {tol:.3E} with final sparsity of {len(u.support)}"
        )
        # Rescale the solution
        for ind, x in enumerate(u.support):
            u.coefficients[ind] /= self.K_norms[x[0]]
        u = u * self.target_norm

        return u


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     target = np.array([1, 0, 4])
#     method = LGCG_finite(M=20, target=target, K=K)
#     method.solve(0.000001)
