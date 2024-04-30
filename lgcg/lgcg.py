import numpy as np
import logging
from typing import Callable
from lib.default_values import *
from lib.ssn import SSN
from lib.lsi import LSI
from lib.measure import Measure

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
        grad_p: Callable,
        hess_p: Callable,
        norm_K_star: float,
        global_search_resolution: float = 10,
        alpha: float = 1,
        gamma: float = 1,
        L: float = 1,
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
        # self.P = lambda u, x: abs(self.p(u)(x))
        self.grad_P = grad_p
        self.hess_P = hess_p
        self.global_search_resolution = global_search_resolution
        self.alpha = alpha
        self.g = get_default_g(self.alpha)
        self.gamma = gamma
        self.L = L
        self.norm_K_star = norm_K_star
        self.Omega = Omega
        self.C = self.L * self.M**2  # TODO
        self.j = lambda u: self.f(u) + self.g(u.coefficients)

    def update_epsilon(self, eta: float, epsilon: float) -> float:
        return (self.M * epsilon + 0.5 * self.C * eta**2) / (self.M + self.M * eta)

    def explicit_Phi(self, p: Callable, u: Measure, v: Measure) -> float:
        # <p(u),v-u>+g(u)-g(v)
        return (
            (v + (u * -1)).duality_pairing(p)
            + self.g(u.coefficients)
            - self.g(v.coefficients)
        )

    def Psi(self, u: Measure, p_u: Callable) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v) over the support of u
        constant_part = -u.duality_pairing(p_u) + self.g(u.coefficients)
        max_on_support = max([abs(p_u(x)) for x in u.support])
        variable_part = max(0, self.M * (max_on_support - self.alpha))
        return constant_part + variable_part

    def lsi(
        self, u: Measure, active_set: np.ndarray, epsilon: float, Psi: float
    ) -> tuple:
        # Implementation of the local support improver
        P = lambda x: abs(self.p(u)(x))
        pass

    def project_into_omega(self, x: np.ndarray) -> np.ndarray:
        # Project a point into Omega
        for dimension, bounds in enumerate(self.Omega):
            x[dimension] = min(max(x[dimension], bounds[0]), bounds[1])
        return x

    def global_search(self, p_u: Callable, u: Measure, epsilon: float) -> tuple:
        P = lambda x: abs(self.p_u(x))
        grad_P = lambda x: self.grad_P(u, x)
        hess_P = lambda x: self.hess_P(u, x)
        grid = (
            np.array(
                np.meshgrid(
                    *(
                        np.linspace(bound[0], bound[1], self.global_search_resolution)
                        for bound in self.Omega
                    )
                )
            )
            .reshape(len(self.Omega), -1)
            .T
        )
        for point in grid:
            # In the future also check for cached points
            if (
                self.explicit_Phi(
                    p_u, u, Measure([point], [self.M * np.sign(p_u(point))])
                )
                >= self.M * epsilon
            ):
                return point, True  # Found a desired point
        processing_array = [
            True for point in grid
        ]  # If the point is still beign optimized
        while any(processing_array):
            for ind, (point, processing) in enumerate(zip(grid, processing_array)):
                if processing:
                    gradient = grad_P(point)
                    if np.linalg.norm(gradient) < 1e-11:
                        processing_array[ind] = False
                        continue
                    hessian = hess_P(point)
                    d = np.linalg.solve(
                        hessian, -gradient
                    )  # TODO deal with singular hessian
                    new_point = self.project_into_omega(
                        point + d
                    )  # TODO: check if it's the right way to project
                    if P(new_point) <= P(point):
                        processing_array[ind] = False
                    else:
                        grid[ind] = new_point
                        if (
                            self.explicit_Phi(
                                p_u,
                                u,
                                Measure(
                                    [new_point], [self.M * np.sign(p_u(new_point))]
                                ),
                            )
                            >= self.M * epsilon
                        ):
                            return new_point, True  # Found a desired point
        values = np.array([P(point) for point in grid])
        arg = np.argmax(values)
        return (
            grid[arg],
            False,
        )  # Found the global maximum, but it does not satisfy the condition

    def build_V(self, p_u: Callable, u: Measure, xi: np.ndarray, old_V: list) -> tuple:
        V = []
        mu = 0
        sign = np.sign(p_u(xi))
        for x, c in zip(u.support, u.coefficients):
            if (
                x not in old_V
                and np.sign(c) == sign
                and np.linalg.norm(x - xi) < 2 * self.R
            ):
                V.append(x)
                mu += abs(c)
        return V, mu

    def local_measure_constructor(
        self, p_u: Callable, u: Measure, x_hat: np.ndarray, lsi_set: list
    ) -> Measure:
        new_support = [x_hat]
        V, mu = self.build_V(p_u, u, x_hat, [])
        new_coefficients = [mu * np.sign(p_u(x_hat))]
        P = lambda x: abs(p_u(x))
        P_lsi = [P(x) for x in lsi_set]
        lsi_set_sorted = [
            x for _, x in sorted(zip(P_lsi, lsi_set), key=lambda t: t[1], reverse=True)
        ]
        for x in lsi_set_sorted:
            V_x, mu_x = self.build_V(p_u, u, x, V)
            if mu_x:
                new_support.append(x)
                new_coefficients.append(mu * np.sign(p_u(x)))
            V += V_x
        # Add unused old supports and coefficients
        for x, c in zip(u.support, u.coefficients):
            if x not in V:
                new_support.append(x)
                new_coefficients.append(c)
        return Measure(new_support, new_coefficients)

    def solve(self, tol: float) -> None:
        u = Measure()
        support_plus = np.array([])
        u_plus = Measure()
        u_plus_hat = Measure()
        p_u = self.p(u)
        epsilon = self.j(u) / self.M
        Psi = epsilon
        k = 1
        while True:
            if k > 1:
                # Low-dimensional step
                if self.j(u_plus) < self.j(u_plus_hat):
                    u_start = Measure(u_plus.support.copy(), u_plus.coefficients.copy())
                else:
                    u_start = Measure(
                        u_plus_hat.support.copy(), u_plus_hat.coefficients.copy()
                    )
                # Insert zero coefficients to unsupported positions
                u_start.add_zero_support(support_plus)
                # Peform SSN
                K_support = np.transpose(np.array([self.k(x) for x in u_start.support]))
                ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
                u_raw = ssn.solve(tol=Psi, u_0=u_start.coefficients)
                # Reconstruct u
                u = Measure(
                    support=u_start.support[u_raw != 0].copy(),
                    coefficients=u_raw[u_raw != 0].copy(),
                )
                p_u = self.p(u)
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            x_hat_lsi, x_check_lsi, x_tilde_lsi = LSI(
                u.support, p_u, epsilon, self.Psi(u, p_u), self.Omega
            )
            x = np.array([])
            if not len(x_tilde_lsi):
                x, valid = self.global_search(p_u, u, epsilon)
            else:
                x = x_hat_lsi
            if not len(x):
                if self.explicit_Phi(p_u, u, Measure()) >= self.M * epsilon:
                    u_plus = u * (1 - eta)
                else:
                    u_plus = u * 1  # Create a new measure with the same parameters
            else:
                v = Measure([x], [self.M * np.sign(p_u(x))])
                u_plus = u * (1 - eta) + v * eta
            if x_hat_lsi:
                v_hat = self.local_measure_constructor(p_u, u, x_hat_lsi, lsi_set)
                u_hat_plus = v_hat * eta + u * (1 - eta)
            # TODO implement the rest. Need LSI to get Phi_k
            k += 1


class LGCG_Finite:
    # An implementation of the LGCG algorithm for finite Omega

    def __init__(
        self,
        M: float,
        target: np.ndarray,
        K: np.ndarray,
        alpha: float = 1,
        gamma: float = 1,
        L: float = 1,
        norm_K_star: float = None,
    ) -> None:
        self.M = M
        self.target = target
        self.K = K
        self.f = get_default_f(self.K, self.target)
        self.p = get_default_p(self.K, self.target)
        self.alpha = alpha
        self.g = get_default_g(self.alpha)
        self.gamma = gamma
        self.L = L
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
        k = 1
        while self.Phi(p_u, u, x) > tol:
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
                np.abs(u_raw) > 1e-11
            ]  # Possibly replace 0 by small precision
            p_u = self.p(u)
            x = np.argmax(np.absolute(p_u))
            k += 1
        logging.info(
            f"LGCG converged in {k} iterations to tolerance {tol} with final sparsity of {len(support)}"
        )
        return {"u": u, "support": support}


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     target = np.array([1, 0, 4])
#     method = LGCG(M=20, target=target, K=K)
#     method.solve(0.000001)
