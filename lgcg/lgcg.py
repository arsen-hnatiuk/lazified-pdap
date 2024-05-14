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
        norm_K_star_L: float,
        norm_K_L: float,
        global_search_resolution: float = 10,
        alpha: float = 1,
        gamma: float = 1,
        theta: float = 1,
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
        self.theta = theta
        self.L = L
        self.norm_K_star = norm_K_star
        self.norm_K_star_L = norm_K_star_L
        self.norm_K_L = norm_K_L
        self.Omega = Omega
        self.C = 4 * self.L * self.M**2 * self.norm_K_L**2
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
        max_P_A = max([abs(p_u(x)) for x in u.support])
        variable_part = max(0, self.M * (max_P_A - self.alpha))
        return constant_part + variable_part

    def lsi(
        self,
        p_u: Callable,
        u: Measure,
        epsilon: float,
        Psi: float,
    ) -> tuple:
        # Implementation of the local support improver
        p_norm = lambda x: abs(p_u(x))
        grad_P = lambda x: self.grad_P(u, x)
        lsi_set = u.support.copy()
        P_on_A = [p_norm(x) for x in u.support]
        max_P_A = max(P_on_A)
        x_hat = np.array([])
        x_hat_value = 0  # Value to maximize by x_hat
        x_tilde = np.array([])
        x_check = np.array([])
        x_check_value = 0  # Value to maximize by x_check
        for ind, point in enumerate(u.support):
            original_point = point
            P_x = p_norm(original_point)
            P_point = P_x
            gradient = grad_P(point)
            condition = (max(0, P_point - max(self.alpha, max_P_A)) + Psi / self.M) / (
                4 * self.R
            )
            while (
                P_point - P_x < 2 * self.R * np.linalg.norm(gradient)
                or np.linalg.norm(gradient) > condition
            ) and self.project_into_omega(point) == point:
                lsi_set[ind] = point
                hessian = np.matmul(np.array([gradient]).T, np.array([gradient]))
                d = np.linalg.solve(
                    hessian + damping * np.eye(len(gradient)), -gradient
                )  # Damped Newton step
                point = point + d
                if np.linalg.norm(point - original_point) >= 2 * self.R:
                    return (
                        x_hat,
                        x_tilde,
                        x_check,
                        lsi_set,
                        False,
                    )  # No maximizer in the given radius
                P_point = p_norm(point)
                gradient = grad_P(point)
                condition = (
                    max(0, P_point - max(self.alpha, max_P_A)) + Psi / self.M
                ) / (4 * self.R)
            if not len(x_tilde):  # Look for point satisfying Phi >= M*epsilon
                while (
                    self.M
                    * (
                        P_point
                        + 2 * self.R * np.linalg.norm(gradient)
                        - max(max_P_A, self.aplha)
                    )
                    + Psi
                    > self.M * epsilon
                ) and self.project_into_omega(point) == point:
                    lsi_set[ind] = point
                    if (
                        self.M * (P_point - max(max_P_A, self.alpha)) + Psi
                        >= self.M * epsilon
                    ):
                        x_tilde = point
                        break
                    hessian = np.matmul(np.array([gradient]).T, np.array([gradient]))
                    d = np.linalg.solve(
                        hessian + damping * np.eye(len(gradient)), -gradient
                    )  # Damped Newton step
                    point = point + d
                    P_point = p_norm(point)
                    gradient = grad_P(point)
                if np.linalg.norm(point - original_point) >= 2 * self.R:
                    return (
                        x_hat,
                        x_tilde,
                        x_check,
                        lsi_set,
                        False,
                    )  # No maximizer in the given radius
            if P_point - P_x > x_hat_value and self.project_into_omega(point) == point:
                x_hat = point
                x_hat_value = P_point - P_x
            if (
                P_point + 2 * self.R * np.linalg.norm(gradient) > x_check_value
                and self.project_into_omega(point) == point
            ):
                x_check = point
                x_check_value = P_point + 2 * self.R * np.linalg.norm(gradient)
        for poin in lsi_set:
            if self.project_into_omega(poin) != point:
                return (
                    x_hat,
                    x_tilde,
                    x_check,
                    lsi_set,
                    False,
                )  # Points ouside of the desired domain
        if (
            self.M * (p_norm(x_hat) - max(max_P_A, self.alpha)) + Psi
            >= self.M * epsilon
        ):
            x_tilde = x_hat
        return x_hat, x_tilde, x_check, lsi_set, True

    def project_into_omega(self, x: np.ndarray) -> np.ndarray:
        # Project a point into Omega
        for dimension, bounds in enumerate(self.Omega):
            x[dimension] = min(max(x[dimension], bounds[0]), bounds[1])
        return x

    def global_search(self, p_u: Callable, u: Measure, epsilon: float) -> tuple:
        p_norm = lambda x: abs(p_u(x))
        grad_P = lambda x: self.grad_P(u, x)
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
                    hessian = np.matmul(np.array([gradient]).T, np.array([gradient]))
                    d = np.linalg.solve(
                        hessian + damping * np.eye(len(gradient)), -gradient
                    )  # Damped Newton step
                    new_point = self.project_into_omega(point + d)
                    if p_norm(new_point) <= p_norm(point):
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
        values = np.array([p_norm(point) for point in grid])
        ind = np.argmax(values)
        return (
            grid[ind],
            False,
        )  # Found the global maximum, but it does not satisfy the condition

    def build_V(self, p_u: Callable, u: Measure, xi: np.ndarray, old_V: list) -> tuple:
        V_local = []
        mu = 0
        sign = np.sign(p_u(xi))
        for x, c in zip(u.support, u.coefficients):
            if (
                x not in old_V
                and np.sign(c) == sign
                and np.linalg.norm(x - xi) < 2 * self.R
            ):
                V_local.append(x)
                mu += abs(c)
        return V_local, mu

    def local_measure_constructor(
        self, p_u: Callable, u: Measure, x_hat: np.ndarray, lsi_set: list
    ) -> Measure:
        new_support = [x_hat]
        all_V, mu = self.build_V(p_u, u, x_hat, [])
        new_coefficients = [mu * np.sign(p_u(x_hat))]
        p_norm = lambda x: abs(p_u(x))
        P_lsi = [p_norm(x) for x in lsi_set]
        lsi_set_sorted = [
            x for _, x in sorted(zip(P_lsi, lsi_set), key=lambda t: t[0], reverse=True)
        ]
        for x in lsi_set_sorted:
            V_x, mu_x = self.build_V(p_u, u, x, all_V)
            if mu_x:
                new_support.append(x)
                new_coefficients.append(mu * np.sign(p_u(x)))
            all_V += V_x
        # Add unused old supports and coefficients
        for x, c in zip(u.support, u.coefficients):
            if x not in all_V:
                new_support.append(x)
                new_coefficients.append(c)
        return Measure(new_support, new_coefficients)

    def solve(self, tol: float) -> None:
        u = Measure()
        support_plus = np.array([])
        u_plus = Measure()
        u_plus_hat = Measure()
        p_u = self.p(u)
        max_P_A = max([abs(p_u(x)) for x in u.support])
        true_Psi = self.Psi(u, p_u)
        epsilon = self.j(u) / self.M
        Psi_k = epsilon
        Phi_k = 1
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
                u_raw = ssn.solve(tol=Psi_k, u_0=u_start.coefficients)
                # Reconstruct u
                u = Measure(
                    support=u_start.support[u_raw != 0].copy(),
                    coefficients=u_raw[u_raw != 0].copy(),
                )
                p_u = self.p(u)
                max_P_A = max([abs(p_u(x)) for x in u.support])
                true_Psi = self.Psi(u, p_u)
            if Phi_k <= tol:
                break
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            x_hat_lsi, x_check_lsi, x_tilde_lsi, lsi_set, lsi_valid = self.lsi(
                p_u, u, epsilon, true_Psi
            )
            x_k = np.array([])
            if not len(x_tilde_lsi):
                x_k, global_valid = self.global_search(p_u, u, epsilon)
            else:
                x_k = x_tilde_lsi
            if not (len(x_tilde_lsi) or global_valid):
                if self.explicit_Phi(p_u, u, Measure()) >= self.M * epsilon:
                    u_plus = u * (1 - eta)
                else:
                    u_plus = u * 1  # Create a new measure with the same parameters
            else:
                v = Measure([x_k], [self.M * np.sign(p_u(x_k))])
                u_plus = u * (1 - eta) + v * eta
            if lsi_valid:
                v_hat = self.local_measure_constructor(p_u, u, x_hat_lsi, lsi_set)
                u_plus_hat = v_hat * eta + u * (1 - eta)
            else:
                u_plus_hat = u_plus * 1  # Create a new measure with the same parameters
            # Build Phi_k
            if lsi_valid:
                Phi_k = (
                    self.M
                    * (
                        max(
                            0,
                            abs(p_u(x_check_lsi)) - max(self.alpha, max_P_A),
                        )
                    )
                    + true_Psi
                )
            else:
                if len(x_tilde_lsi) or global_valid:
                    Phi_k = self.M * epsilon
                else:  # x_k is the exact global maximum of P
                    Phi_k = (
                        self.M * (max(0, abs(p_u(x_k)) - max(self.alpha, max_P_A)))
                        + true_Psi
                    )
            constant = (
                8
                * self.M
                * self.L
                * self.norm_K_star
                * max(3, 4 * self.C * self.theta * self.M**2 * self.norm_K_star_L)
                * np.sqrt(Psi_k)
                / np.sqrt(self.gamma)
                + 4 * Psi_k
            )
            if constant > Phi_k:
                # recompute step
                Psi_k = min(Psi_k / 2, Phi_k**2)
                continue
            support_plus = np.unique(
                np.append(
                    u.support.copy(),
                    np.append(u_plus.support.copy(), u_plus_hat.support.copy(), axis=0),
                    axis=0,
                )
            )
            k += 1
        return u


class LGCG_finite:
    # An implementation of the LGCG algorithm for finite Omega

    def __init__(
        self,
        target: np.ndarray,
        K: np.ndarray,
        alpha: float = 1,
    ) -> None:
        self.target = target
        self.M = np.linalg.norm(self.target) ** 2  # value of j at initial u
        self.K = K
        self.f = get_default_f(self.K, self.target)
        self.p = get_default_p(self.K, self.target)
        self.alpha = alpha
        self.g = get_default_g(self.alpha)
        self.L = 1
        self.norm_K = np.max(
            [np.linalg.norm(row) for row in np.transpose(self.K)]
        )  # the 2,inf norm of K* = the 1,2 norm of K
        self.C = 4 * self.L * self.M**2 * self.norm_K**2
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
#     method = LGCG_finite(M=20, target=target, K=K)
#     method.solve(0.000001)
