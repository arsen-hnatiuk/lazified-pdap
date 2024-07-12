import numpy as np
import logging
from typing import Callable
from lib.default_values import *
from lib.ssn import SSN
from lib.measure import Measure

logging.basicConfig(
    level=logging.DEBUG,
)


class LGCG:
    def __init__(
        self,
        target: np.ndarray,
        k: Callable,
        g: Callable,
        f: Callable,
        p: Callable,
        grad_P: Callable,
        hess_P: Callable,
        norm_K_star: float,
        norm_K_star_L: float,
        grad_j: Callable,
        hess_j: Callable,
        alpha: float,
        Omega: np.ndarray,
        gamma: float,
        theta: float,
        sigma: float,
        m: float,
        bar_m: float,
        L: float,
        R: float,
        global_search_resolution: float = 10,
    ) -> None:
        self.target = target
        self.k = k
        self.f = f
        self.p = p
        self.grad_P = grad_P
        self.hess_P = hess_P
        self.global_search_resolution = global_search_resolution
        self.alpha = alpha
        self.g = g
        self.gamma = gamma
        self.theta = theta
        self.sigma = sigma
        self.m = m
        self.bar_m = bar_m
        self.L = L
        self.norm_K_star = norm_K_star
        self.norm_K_star_L = norm_K_star_L
        self.Omega = Omega  # Example [[0,1],[1,2]] for [0,1]x[1,2]
        self.j = lambda u: self.f(u) + self.g(u.coefficients)
        self.u_0 = Measure()
        self.M = min(self.j(self.u_0) / self.alpha, 10)  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2 * self.norm_K_star**2
        self.R = R
        self.grad_j = grad_j
        self.hess_j = hess_j
        self.machine_precision = 1e-12
        self.stop_search = 5

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
        if len(u.support):
            max_P_A = max([abs(p_u(x)) for x in u.support])
        else:
            max_P_A = 0
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
        if not len(u.support):
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                False,
            )  # No points to improve around
        p_norm = lambda x: abs(p_u(x))
        grad_P = lambda x: self.grad_P(x, u)
        hess_P = lambda x: self.hess_P(x, u)
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
            hessian = hess_P(point)
            condition = (max(0, P_point - max(self.alpha, max_P_A)) + Psi / self.M) / (
                4 * self.R
            )
            point_steps = 0
            while (
                P_point - P_x < 2 * self.R * np.linalg.norm(gradient)
                or np.linalg.norm(gradient) > condition
            ) and point_steps < self.stop_search:
                d = np.linalg.solve(hessian, -gradient)  # Newton step
                point = point + d
                if np.linalg.norm(
                    point - original_point
                ) >= 2 * self.R or not np.array_equal(
                    self.project_into_omega(point), point
                ):
                    return (
                        x_hat,
                        x_tilde,
                        x_check,
                        lsi_set,
                        False,
                    )  # No maximizer in the given radius
                lsi_set[ind] = point
                P_point = p_norm(point)
                gradient = grad_P(point)
                hessian = hess_P(point)
                condition = (
                    max(0, P_point - max(self.alpha, max_P_A)) + Psi / self.M
                ) / (4 * self.R)
                point_steps += 1
            if not len(x_tilde):  # Look for point satisfying Phi >= M*epsilon
                point_steps = 0
                while (
                    self.M
                    * (
                        P_point
                        + 2 * self.R * np.linalg.norm(gradient)
                        - max(max_P_A, self.alpha)
                    )
                    + Psi
                    > self.M * epsilon
                ) and point_steps < self.stop_search:
                    if (
                        self.M * (P_point - max(max_P_A, self.alpha)) + Psi
                        >= self.M * epsilon
                    ):
                        x_tilde = point
                        break
                    d = np.linalg.solve(hessian, -gradient)  # Newton step
                    point = point + d
                    if np.linalg.norm(
                        point - original_point
                    ) >= 2 * self.R or not np.array_equal(
                        self.project_into_omega(point), point
                    ):
                        return (
                            x_hat,
                            x_tilde,
                            x_check,
                            lsi_set,
                            False,
                        )  # No maximizer in the given radius
                    lsi_set[ind] = point
                    P_point = p_norm(point)
                    gradient = grad_P(point)
                    hessian = hess_P(point)
                    point_steps += 1
            if P_point - P_x > x_hat_value:
                x_hat = point
                x_hat_value = P_point - P_x
            if P_point + 2 * self.R * np.linalg.norm(gradient) > x_check_value:
                x_check = point
                x_check_value = P_point + 2 * self.R * np.linalg.norm(gradient)
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

    def global_search(self, u: Measure, epsilon: float) -> tuple:
        p_u = self.p(u)
        p_norm = lambda x: abs(p_u(x))
        grad_P = lambda x: self.grad_P(x, u)
        hess_P = lambda x: self.hess_P(x, u)
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
        point_steps = 0
        while any(processing_array) and point_steps < self.stop_search:
            for ind, (point, processing) in enumerate(zip(grid, processing_array)):
                if processing:
                    gradient = grad_P(point)
                    hessian = hess_P(point)
                    if np.linalg.norm(gradient) < self.machine_precision:
                        processing_array[ind] = False
                        continue
                    d = np.linalg.solve(hessian, -gradient)  # Newton step
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
            point_steps += 1
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
                all([not np.array_equal(x, arr) for arr in old_V])
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
            if all([not np.array_equal(x, arr) for arr in all_V]):
                new_support.append(x)
                new_coefficients.append(c)
        return Measure(new_support, new_coefficients)

    def solve(self, tol: float) -> tuple:
        u = Measure()
        u_plus_tilde = Measure()
        u_plus_hat = Measure()
        p_u = self.p(u)
        max_P_A = 0
        true_Psi = self.Psi(u, p_u)
        epsilon = self.j(u) / self.M
        Psi_k = self.gamma * self.sigma / (5 * self.norm_K_star**2 * self.L**2)
        Phi_k = 1
        choice = "N/A"
        k = 1
        s = 1
        Phi_ks = [Phi_k]
        objective_values = [self.j(u)]
        steps = []
        while Phi_k > tol:
            if k > 1:
                # Low-dimensional step
                if self.j(u_plus_tilde) < self.j(u_plus_hat):
                    u_plus = u_plus_tilde * 1
                    choice = "GCG"
                else:
                    u_plus = u_plus_hat * 1
                    choice = "LSI"
                # Peform SSN
                K_support = np.transpose(np.array([self.k(x) for x in u_plus.support]))
                ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
                u_raw = ssn.solve(
                    tol=max(Psi_k, self.machine_precision), u_0=u_plus.coefficients
                )
                u_raw[np.abs(u_raw) < self.machine_precision] = 0
                # Reconstruct u
                u = Measure(
                    support=u_plus.support[u_raw != 0].copy(),
                    coefficients=u_raw[u_raw != 0].copy(),
                )
                p_u = self.p(u)
                max_P_A = max([abs(p_u(x)) for x in u.support])
                true_Psi = self.Psi(u, p_u)

            # Find new support points
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)
            x_hat_lsi, x_tilde_lsi, x_check_lsi, lsi_set, lsi_valid = self.lsi(
                p_u, u, epsilon, true_Psi
            )
            x_k = np.array([])
            if not len(x_tilde_lsi):
                x_k, global_valid = self.global_search(u, epsilon)
                # TODO implement stopping for global search e.g. if u_hat chosen last iteration
            else:
                x_k = x_tilde_lsi

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

            # Prepare and check for recompute
            constant = (
                12
                * self.M
                * self.L
                * self.norm_K_star
                * np.sqrt(Psi_k)
                / np.sqrt(self.gamma)
                + 2 * Psi_k
            )
            if constant > Phi_k and Psi_k > self.machine_precision:
                # Recompute step and updae running metrics
                Psi_k = min(Psi_k / 2, Phi_k**2)
                logging.info(
                    f"Recompute {s}, Psi_k:{Psi_k:.3E}, Phi_k:{Phi_k:.3E}, constant:{constant:.3E}"
                )
                Phi_ks.append(Phi_k)
                objective_values.append(self.j(u))
                steps.append("recompute")
                s += 1
                continue

            # Normal step, construct new iterate
            v = Measure([x_k], [self.M * np.sign(p_u(x_k))])
            if not (len(x_tilde_lsi) or global_valid):
                if self.explicit_Phi(p_u, u, Measure()) >= self.M * epsilon:
                    u_plus_tilde = u * (1 - eta)
                elif self.explicit_Phi(p_u, u, v) >= 0:
                    eta_local = self.explicit_Phi(p_u, u, v) / self.C
                    u_plus_tilde = u * (1 - eta_local) + v * eta_local
                else:
                    u_plus_tilde = (
                        u * 1
                    )  # Create a new measure with the same parameters
            else:
                u_plus_tilde = u * (1 - eta) + v * eta
            if lsi_valid:
                v_hat = self.local_measure_constructor(p_u, u, x_hat_lsi, lsi_set)
                nu_constant = 4 * (
                    32
                    * self.L
                    * self.M**2
                    * self.norm_K_star_L**2
                    * (1 + self.L * self.norm_K_star / (np.sqrt(self.gamma)))
                    / self.theta
                )
                nu = min(1, abs(v_hat.coefficients[0]) / nu_constant)
                u_plus_hat = u * (1 - nu) + v_hat * nu
            else:
                u_plus_hat = (
                    u_plus_tilde * 1
                )  # Create a new measure with the same parameters

            # Update running metrics
            Phi_ks.append(Phi_k)
            objective_values.append(self.j(u))
            steps.append("normal")
            logging.info(f"{k}: {choice}, Phi_k: {Phi_k}, support: {u.support}")
            k += 1
        return u, Phi_ks, objective_values

    def cluster_exact(self, u, radius) -> Measure:
        if not len(u.coefficients):
            return u
        new_support = []
        new_coefs = []
        for i, point in enumerate(u.support):
            added = False
            for j, other_point in enumerate(new_support):
                if np.linalg.norm(point - other_point) < radius:
                    new_support[j] = 0.5 * (point + other_point)
                    new_coefs[j] = new_coefs[j] + u.coefficients[i]
                    added = True
                    break
            if not added:
                new_support.append(point)
                new_coefs.append(u.coefficients[i])
        return Measure(new_support, new_coefs)

    def solve_exact(self, tol: float) -> tuple:
        k = 1
        u = Measure()
        p_u = self.p(u)
        x, global_valid = self.global_search(u, 1000)
        P_value = np.abs(p_u(x))
        P_values = [P_value]
        objective_values = [self.j(u)]
        while (
            len(u.coefficients) == 0
            or P_value < self.alpha
            or P_value - self.alpha > tol
        ):
            eta = 4 / (k + 3)
            v = Measure(support=np.array([x]), coefficients=[1])
            u_plus = u * (1 - eta) + v * eta
            K_support = np.transpose(np.array([self.k(x) for x in u_plus.support]))
            ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
            u_raw = ssn.solve(tol=self.machine_precision, u_0=u_plus.coefficients)
            u_raw[np.abs(u_raw) < self.machine_precision] = 0
            u = Measure(
                support=u_plus.support[u_raw != 0].copy(),
                coefficients=u_raw[u_raw != 0].copy(),
            )
            # u = self.cluster_exact(u, 1e-4)
            p_u = self.p(u)
            x, global_valid = self.global_search(u, 1000)
            P_value = np.abs(p_u(x))
            logging.info(
                f"{k}: P_value:{P_value-self.alpha:.3E}, support: {u.support}, coefs: {u.coefficients}, x: {x}, valid: {global_valid}"
            )
            logging.info("==============================================")
            k += 1
            P_values.append(P_value)
            objective_values.append(self.j(u))
            # if k == 25:
            #     return p_u, u, P_values, objective_values
        return u, P_values, objective_values

    def local_clustering(self, u: Measure, p_u: Callable) -> tuple:
        sorting_values = [0] * len(u.support)
        for ind, (x, c) in enumerate(zip(u.support, u.coefficients)):
            for x_local, c_local in zip(u.support, u.coefficients):
                if np.linalg.norm(x - x_local) < 2 * self.R:
                    sorting_values[ind] += c_local * p_u(x_local)
        tuples = [(x, c) for x, c in zip(u.support, u.coefficients)]
        tuples_sorted = [
            x
            for _, x in sorted(
                zip(sorting_values, tuples), key=lambda t: t[0], reverse=True
            )
        ]
        clustered_tuples = []
        already_clustered = []
        for x, c in tuples_sorted:
            if not len(already_clustered) or not np.any(
                np.all(np.array(already_clustered) == x, axis=1)
            ):
                coefficient = 0
                for x_local, c_local in tuples_sorted:
                    if (
                        not len(already_clustered)
                        or not np.any(
                            np.all(np.array(already_clustered) == x_local, axis=1)
                        )
                    ) and np.linalg.norm(x - x_local) < 2 * self.R:
                        coefficient += c_local
                        already_clustered.append(x_local)
                clustered_tuples.append((x, coefficient))
        return clustered_tuples

    def compute_newton_step(
        self, points: np.ndarray, coefs: np.ndarray, damped: bool
    ) -> Measure:
        grad_j_z = self.grad_j(points, coefs)
        hess_j_z = self.hess_j(points, coefs)
        if damped:
            nu = min(
                1,
                (
                    -self.m
                    + np.sqrt(
                        self.m**2
                        + 8 * self.m * self.L * self.bar_m**3 * np.linalg.norm(grad_j_z)
                    )
                )
                / (4 * self.L * self.bar_m**3 * np.linalg.norm(grad_j_z)),
            )
        else:
            nu = 1
        update_direction = np.linalg.solve(hess_j_z, grad_j_z)
        # Transform vector into tuples
        coefs -= nu * update_direction[-len(coefs) :]
        for i, point in enumerate(points):
            new_point = point - (
                nu
                * update_direction[
                    i * self.Omega.shape[0] : (i + 1) * self.Omega.shape[0]
                ]
            )
            points[i] = self.project_into_omega(new_point)
        u_plus = Measure(points, coefs)
        return u_plus

    def solve_newton(self, tol: float, clustering_frequency=5, damped=True) -> tuple:
        u = Measure()
        p_u = self.p(u)
        epsilon = self.j(u) / self.M
        Psi_1 = self.gamma * self.sigma / (5 * self.norm_K_star**2 * self.L**2)
        Psi_k = Psi_1
        k = 1
        steps_since_clustering = 0
        last_value_clustering = self.j(u)
        grad_norm = 1
        grad_norms = [grad_norm]
        objective_values = [self.j(u)]
        while grad_norm > max(tol, tol / self.m):
            # Newton step
            if len(u.coefficients):
                u_tilde_plus = self.compute_newton_step(c_points, c_coefs, damped)
            else:
                u_tilde_plus = u

            # GCG step
            if grad_norm > 1e-8:
                eta = 4 / (k + 3)
                epsilon = self.update_epsilon(eta, epsilon)
                x_k, global_valid = self.global_search(u, epsilon)
                v = Measure([x_k], [self.M * np.sign(p_u(x_k))])
                if not global_valid:
                    if self.explicit_Phi(p_u, u, Measure()) >= self.M * epsilon:
                        u_hat_plus = u * (1 - eta)
                    elif self.explicit_Phi(p_u, u, v) >= 0:
                        eta_local = self.explicit_Phi(p_u, u, v) / self.C
                        u_hat_plus = u * (1 - eta_local) + v * eta_local
                    else:
                        u_hat_plus = (
                            u * 1
                        )  # Create a new measure with the same parameters
                else:
                    u_hat_plus = u * (1 - eta) + v * eta
            else:
                u_hat_plus = u * 1

            if (
                self.j(u_hat_plus) > self.j(u_tilde_plus)
                or abs(self.j(u_hat_plus) - self.j(u_tilde_plus))
                < self.machine_precision
            ):
                choice = "newton"
                u_plus = u_tilde_plus
                local_Psi = Psi_1
                steps_since_clustering = 0
            else:
                choice = "gcg"
                u_plus = u_hat_plus
                local_Psi = Psi_k
                Psi_k = max(Psi_k / 2, self.machine_precision)
                steps_since_clustering += 1

            # Low-dimensional step
            K_support = np.transpose(np.array([self.k(x) for x in u_plus.support]))
            ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
            u_raw = ssn.solve(tol=local_Psi, u_0=u_plus.coefficients)
            u_raw[np.abs(u_raw) < self.machine_precision] = 0
            # Reconstruct u
            u = Measure(
                support=u_plus.support[u_raw != 0].copy(),
                coefficients=u_raw[u_raw != 0].copy(),
            )
            p_u = self.p(u)
            if choice == "newton":
                last_value_clustering = self.j(u)

            if len(u.coefficients):
                c_points, c_coefs = tuple(zip(*self.local_clustering(u, p_u)))
                c_points = list(c_points)
                c_coefs = list(c_coefs)
                grad_j_z = self.grad_j(c_points, c_coefs)
                grad_norm = np.linalg.norm(grad_j_z)
                if (
                    self.j(u) < last_value_clustering
                    and steps_since_clustering > clustering_frequency
                ):
                    # Force clustering
                    last_value_clustering = self.j(u)
                    steps_since_clustering = 0
                    u = Measure(c_points, c_coefs)

            logging.info(
                f"{k}: {choice}, support: {u.support}, coefs: {u.coefficients}, grad_norm:{grad_norm:.3E}, objective: {self.j(u):.3E}"
            )
            k += 1
            grad_norms.append(grad_norm)
            objective_values.append(self.j(u))

        # Last clustering
        c_points, c_coefs = tuple(zip(*self.local_clustering(u, p_u)))
        u = Measure(c_points, c_coefs)
        return u, grad_norms, objective_values
