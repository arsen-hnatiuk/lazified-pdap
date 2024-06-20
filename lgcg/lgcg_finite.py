import numpy as np
import logging
import time
from lib.default_values import *
from lib.ssn import SSN

logging.basicConfig(
    level=logging.DEBUG,
)


class LGCG_finite:
    # An implementation of the LGCG algorithm for finite Omega

    def __init__(
        self,
        target: np.ndarray,
        K: np.ndarray,
        alpha: float = 1,
    ) -> None:
        K = np.append(
            K,
            np.ones((K.shape[0], 1)),
            axis=1,
        )  # Adding a constant term
        self.K_norms = np.linalg.norm(K, axis=0)
        self.K_norms[self.K_norms == 0] = 1  # Avoid division by zero
        self.K_transpose = np.array(
            [row / norm for row, norm in zip(K.T, self.K_norms)]
        )
        self.K = np.transpose(self.K_transpose)
        self.target_norm = np.linalg.norm(target)
        self.target = target / self.target_norm
        self.f = get_default_f(self.K, self.target)
        self.p = get_default_p(self.K, self.target)
        self.alpha = alpha
        self.g = get_default_g(self.alpha)
        self.L = 1
        self.norm_K = np.max(
            [np.linalg.norm(row) for row in np.transpose(self.K)]
        )  # the 2,inf norm of K* = the 1,2 norm of K
        self.u_0 = np.eye(1, self.K.shape[1], self.K.shape[1] - 1)[
            0
        ]  # Only the constatn term
        self.j = lambda u: self.f(u) + self.g(u)
        self.M = self.j(self.u_0) / self.alpha  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2 * self.norm_K**2  # Smoothness constant
        self.machine_precision = 1e-11

    def update_epsilon(self, eta: float, epsilon: float) -> float:
        return (self.M * epsilon + 0.5 * self.C * eta**2) / (self.M + self.M * eta)

    def explicit_Phi(self, p: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        # <p(u),v-u>+g(u)-g(v)
        return np.matmul(p, v - u) + self.g(u) - self.g(v)

    def Phi(self, p_u: np.ndarray, u: np.ndarray, x: int) -> float:
        # M*max{0,||p_u||-alpha}+g(u)-<p_u,u>
        return (
            self.M * (max(0, np.absolute(p_u[x]) - self.alpha))
            + self.g(u)
            - np.matmul(p_u, u)
        )

    def solve(self, tol: float) -> dict:
        u = self.u_0
        support = np.where(u != 0)[0]
        p_u = self.p(u)
        x = np.argmax(np.absolute(p_u))
        epsilon = self.j(u) / self.M
        Psi = epsilon
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        ssn_time = 0
        while Phi_value > tol:
            u_old = u.copy()
            Psi_old = Psi
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            Psi = max(min(Psi, self.M * epsilon), self.machine_precision)
            if x in support:
                Psi = Psi / 2
            v = self.M * np.sign(p_u[x]) * np.eye(1, self.K.shape[1], x)[0]
            Phi_x = self.explicit_Phi(p=p_u, u=u, v=v)
            if Phi_x >= self.M * epsilon:
                u = (1 - eta) * u + eta * v
            elif (
                self.explicit_Phi(p=p_u, u=u, v=np.zeros(self.K.shape[1]))
                >= self.M * epsilon
            ):
                u = (1 - eta) * u
            elif Phi_x > 0:
                eta_local = Phi_x / self.C
                u = (1 - eta_local) * u + eta_local * v

            if not np.array_equal(u, u_old) or Psi_old != Psi:
                # Low-dimensional optimization
                ssn_start = time.time()
                support_extended = np.where(u != 0)[0]
                K_support = self.K[:, support_extended]
                ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
                u_raw = ssn.solve(tol=Psi, u_0=u[support_extended])
                u_raw[np.abs(u_raw) < self.machine_precision] = 0
                ssn_time += time.time() - ssn_start

                if not np.array_equal(u_raw, u_old[support_extended]):
                    # SSN found a different solution
                    u = np.zeros(len(u))
                    for ind, pos in enumerate(support_extended):
                        u[pos] = u_raw[ind]
                    p_u = self.p(u)
                    x = np.argmax(np.absolute(p_u))
                    Phi_value = self.Phi(p_u, u, x)
                    support = np.where(u != 0)[0]

            logging.info(
                f"{k}: Phi {Phi_value:.3E}, epsilon {epsilon:.3E}, support {support}, Psi {Psi:.3E}"
            )
            k += 1
        logging.info(
            f"LGCG converged in {k} iterations and {time.time()-start_time:.3f}s (SSN time {ssn_time:.3f}s) to tolerance {tol:.3E} with final sparsity of {len(support)}"
        )
        # Rescale the solution
        for ind, pos in enumerate(self.K_norms):
            u[ind] /= pos
        u = u * self.target_norm

        return {"u": u[support], "support": support}

    def solve_exact(self, tol: float) -> dict:
        u = self.u_0
        support = np.where(u != 0)[0]
        p_u = self.p(u)
        x = np.argmax(np.absolute(p_u))
        k = 1
        Phi_value = self.Phi(p_u, u, x)
        start_time = time.time()
        ssn_time = 0
        while Phi_value > tol:
            eta = 4 / (k + 4)
            v = self.M * np.sign(p_u[x]) * np.eye(1, self.K.shape[1], x)[0]
            u = (1 - eta) * u + eta * v

            # Low-dimensional optimization
            support_extended = np.where(u != 0)[0]
            K_support = self.K[:, support_extended]
            ssn_start = time.time()
            ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
            u_raw = ssn.solve(tol=self.machine_precision, u_0=u[support_extended])
            u_raw[np.abs(u_raw) < self.machine_precision] = 0
            ssn_time += time.time() - ssn_start

            u = np.zeros(len(u))
            for ind, pos in enumerate(support_extended):
                u[pos] = u_raw[ind]
            p_u = self.p(u)
            x = np.argmax(np.absolute(p_u))
            Phi_value = self.Phi(p_u, u, x)
            support = np.where(u != 0)[0]

            logging.info(f"{k}: Phi {Phi_value:.3E}, support {support}")
            k += 1
        logging.info(
            f"LGCG converged in {k} iterations and {time.time()-start_time:.3f}s (SSN time {ssn_time:.3f}s) to tolerance {tol:.3E} with final sparsity of {len(support)}"
        )
        # Rescale the solution
        for ind, pos in enumerate(self.K_norms):
            u[ind] /= pos
        u = u * self.target_norm

        return {"u": u[support], "support": support}


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     target = np.array([1, 0, 4])
#     method = LGCG_finite(M=20, target=target, K=K)
#     method.solve(0.000001)
