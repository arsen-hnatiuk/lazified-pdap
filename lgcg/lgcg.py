import numpy as np
import logging
from typing import Callable
import copy
from lib.default_values import *
from lib.ssn import SSN
from lib.lsi import LSI
from lib.measure import Measure, add_measures

logging.basicConfig(
    level=logging.DEBUG,
)


class LGCG:
    def __init__(
        self,
        M: float,
        target: np.ndarray,
        k: Callable,
        p: Callable,
        alpha: float = None,
        gamma: float = None,
        L: float = None,
        norm_K_star: float = None,
        Omega: np.ndarray = None,
    ) -> None:
        self.M = M
        self.target = target
        self.k = k
        self.f = (
            lambda u: 0.5
            * np.linalg.norm(
                np.sum(
                    np.array(
                        [c * self.k(x) for x, c in zip(u.support, u.coefficients)]
                    ),
                    axis=0,
                )
                - self.target
            )
            ** 2
        )
        self.p = p
        self.alpha = alpha if not alpha is None else 1
        self.g = get_default_g(self.alpha)
        self.gamma = gamma if not gamma is None else 1
        self.L = L if not L is None else 1
        self.norm_K_star = (
            norm_K_star
            if not norm_K_star is None
            else np.max([np.linalg.norm(row) for row in np.transpose(self.K)])
        )  # the 2,inf norm of the transpose of K
        self.Omega = Omega
        self.C = self.L * self.M**2
        self.j = lambda u: self.f(u) + self.g(u.coefficients)

    def update_epsilon(self, eta: float, epsilon: float) -> float:
        return (self.M * epsilon + 0.5 * self.C * eta**2) / (self.M + self.M * eta)

    def explicit_Phi(self, p: Callable, u: Measure, v: Measure) -> float:
        # <p(u),v-u>+g(u)-g(v)
        u = u.copy()  # TODO
        return (
            add_measures(v, u.multiply(-1)).duality_pairing(p)
            + self.g(u.coefficients)
            - self.g(v.coefficients)
        )

    def Psi(self, u: Measure, p_u: Callable) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v) over the support of u
        constant_part = -u.duality_pairing(p_u) + self.g(u.coefficients)
        max_on_support = max([abs(p_u(x)) for x in u.support])
        variable_part = max(0, self.M * (max_on_support - self.alpha))
        return constant_part + variable_part

    def global_search(self, p: Callable, epsilon: float) -> np.ndarray:
        # TODO
        pass

    def local_measure_constructor(self, u, x_hat, x_check, x_tilde) -> Measure:
        # TODO
        pass

    def solve(self, tol: float) -> None:
        u = Measure()
        support_plus = np.array([])
        u_plus = Measure()
        u_plus_hat = Measure()
        p_u = self.p(u)
        epsilon = self.j(u) / self.M
        Psi = epsilon
        k = 0
        while True:
            if len(support_plus):
                # Low-dimensional step
                K_support = np.transpose(np.array([self.k(x) for x in support_plus]))
                if self.j(u_plus) < self.j(u_plus_hat):
                    u_start = u_plus
                else:
                    u_start = u_plus_hat
                # Insert zero coefficients to unsupported positions
                u_start.add_zero_support(support_plus)
                # Peform SSN
                ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
                u_raw = ssn.solve(tol=Psi, u_0=u_start)
                # Reconstruct u
                u = Measure(
                    support=u_start.support[u_raw != 0],
                    coefficients=u_raw[u_raw != 0],
                )
                p_u = self.p(u)
            k += 1
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            x_hat_lsi, x_check_lsi, x_tilde_lsi = LSI(
                u.support, p_u, epsilon, self.Psi(u, p_u), self.Omega
            )
            if not x_tilde_lsi:
                x = self.global_search(p_u, epsilon)
            else:
                x = x_hat_lsi
            if not x:
                if self.explicit_Phi(u, Measure()) >= self.M * epsilon:
                    u_plus = u.multiply(1 - eta)
                else:
                    u_plus = u
            else:
                v = Measure(
                    support=np.array([x]), coefficients=np.array([self.M])
                )  # TODO mult by sgn p(x)
                u_plus = add_measures(u.multiply(1 - eta), v.multiply(eta))
            if x_hat_lsi:
                v_hat = self.local_measure_constructor(
                    u, x_hat_lsi, x_check_lsi, x_tilde_lsi
                )
                u_hat_plus = add_measures(v_hat.multiply(eta), u.multiply(1 - eta))
            # TODO implement the rest. Need LSI to get Phi_k


class LGCG_Finite:
    # An implementation of the LGCG algorithm for finite Omega

    def __init__(
        self,
        M: float,
        target: np.ndarray,
        K: np.ndarray,
        alpha: float = None,
        gamma: float = None,
        L: float = None,
        norm_K_star: float = None,
    ) -> None:
        self.M = M
        self.target = target
        self.K = K
        self.f = get_default_f(self.K, self.target)
        self.p = get_default_p(self.K, self.target)
        self.alpha = alpha if not alpha is None else 1
        self.g = get_default_g(self.alpha)
        self.gamma = gamma if not gamma is None else 1
        self.L = L if not L is None else 1
        self.norm_K_star = (
            norm_K_star
            if not norm_K_star is None
            else np.max([np.linalg.norm(row) for row in np.transpose(self.K)])
        )  # the 2,inf norm of the transpose of K
        self.C = self.L * self.M**2
        self.j = lambda u: self.f(u) + self.g(u)

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
        support = np.array([])
        u = np.zeros(self.K.shape[1])
        p_u = self.p(u)
        x = np.argmax(np.absolute(p_u))
        epsilon = self.j(u) / self.M
        Psi = epsilon
        k = 0
        while self.Phi(p_u, u, x) > tol:
            k += 1
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            Psi = min(Psi, self.M * epsilon)
            if x in support:
                support_extended = support
                Psi = Psi / 2
            else:
                support_extended = np.unique(
                    np.append(support, x).astype(int)
                )  # returns sorted
            v = self.M * np.sign(p_u[x]) * np.eye(1, self.K.shape[1], x)[0]
            if self.explicit_Phi(p=p_u, u=u, v=v) >= self.M * epsilon:
                u = (1 - eta) * u + eta * v
            elif (
                self.explicit_Phi(p=p_u, u=u, v=np.zeros(self.K.shape[1]))
                >= self.M * epsilon
            ):
                u = (1 - eta) * u

            # Low-dimensional step
            K_support = self.K[:, support_extended]
            ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
            u_raw = ssn.solve(tol=Psi, u_0=u[support_extended])

            u = np.zeros(len(u))
            for ind, pos in enumerate(support_extended):
                u[pos] = u_raw[ind]
            support = support_extended[
                u_raw != 0
            ]  # Possibly replace 0 by small precision
            p_u = self.p(u)
            x = np.argmax(np.absolute(p_u))
        logging.info(
            f"LGCG converged in {k} iterations to tolerance {tol} with final sparsity of {len(support)}"
        )
        return {"u": u, "support": support}


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     target = np.array([1, 0, 4])
#     method = LGCG(M=20, target=target, K=K)
#     method.solve(0.000001)
