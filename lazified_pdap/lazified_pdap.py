import numpy as np
import logging
import time
from typing import Callable
from sklearn.utils import gen_batches
from lib.default_values import *
from lib.ssn import SSN
from lib.measure import Measure

logging.basicConfig(
    level=logging.DEBUG,
)


class LazifiedPDAP:
    def __init__(
        self,
        target: np.ndarray,
        kappa: Callable,
        g: Callable,
        f: Callable,
        p: Callable,
        grad_P: Callable,
        hess_P: Callable,
        norm_kappa: float,
        norm_kappa1: float,
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
        L_H: float,
        R: float,
        projection: str = "box",  # box or sphere
        M: int = 1e6,
        random_grid_size: int = int(1e4),
    ) -> None:
        self.target = target
        self.kappa = kappa
        self.f = f
        self.p = p
        self.grad_P = grad_P
        self.hess_P = hess_P
        self.alpha = alpha
        self.g = g
        self.gamma = gamma
        self.theta = theta
        self.sigma = sigma
        self.m = m
        self.bar_m = bar_m
        self.L = L
        self.L_H = L_H
        self.norm_kappa = norm_kappa
        self.norm_kappa1 = norm_kappa1
        self.Omega = Omega  # Example [[0,1],[1,2]] for [0,1]x[1,2]
        self.j = lambda u: self.f(u) + self.g(u.coefficients)
        self.j_tilde = lambda pos, coef: self.f(Measure(pos, coef)) + self.g(coef)
        self.u_0 = Measure()
        self.M = min(M, self.j(self.u_0) / self.alpha)  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2 * self.norm_kappa**2
        self.R = R
        self.projection = projection
        self.global_search_resolution = int(
            max([bound[1] - bound[0] for bound in self.Omega]) / self.R
        )
        self.grad_j = grad_j
        self.hess_j = hess_j
        self.Psi_0 = 1e-3
        self.machine_precision = 1e-12
        self.stop_search = 5
        self.batching_constant = 2e8
        self.random_grid_size = random_grid_size

    def project_into_domain(self, x: np.ndarray) -> np.ndarray:
        # Project an array into domain, parallelized
        if self.projection == "sphere":
            norms = np.linalg.norm(x, axis=1)
            return np.divide(x, norms.reshape(-1, 1))
        else:
            for i, bounds in zip(range(x.shape[1]), self.Omega):
                column = x[:, i].copy()
                x[:, i] = np.clip(column, bounds[0], bounds[1])
            return x

    # def lsi(
    #     self,
    #     p_u: Callable,
    #     u: Measure,
    #     epsilon: float,
    #     Phi_A: float,
    # ) -> tuple:
    #     # Implementation of the local support improver
    #     if not len(u.support):
    #         return (
    #             np.array([]),
    #             np.array([]),
    #             np.array([]),
    #             np.array([]),
    #         )  # No points to improve around
    #     q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
    #     p_norm = lambda x: np.abs(p_u(x))
    #     grad_P = self.grad_P(u)
    #     hess_P = self.hess_P(u)
    #     lsi_set = u.support.copy()

    #     max_P_A = np.max(p_norm(lsi_set))
    #     original_P_vals = p_norm(lsi_set)
    #     processing_array = np.array([True for _ in range(len(lsi_set))])
    #     P_vals = original_P_vals
    #     gradients = grad_P(lsi_set)
    #     gradients_norms = np.linalg.norm(gradients, axis=1)

    #     point_steps = 0
    #     while point_steps < self.stop_search:
    #         condition_2_rhs = np.maximum(
    #             (np.maximum(P_vals - max(self.alpha, max_P_A), 0) + Phi_A / self.M)
    #             / (4 * self.R),
    #             self.machine_precision,
    #         )
    #         condition_1 = P_vals - original_P_vals < 2 * self.R * gradients_norms
    #         condition_2 = gradients_norms > condition_2_rhs
    #         processing_array = condition_1 | condition_2  # element-wise or

    #         if not any(processing_array):
    #             break

    #         active_indices = np.where(processing_array)[0]
    #         active_points = lsi_set[active_indices]
    #         new_points = np.zeros(active_points.shape)
    #         active_gradients = gradients[active_indices]
    #         hessians = (
    #             active_gradients
    #             if self.projection == "sphere"
    #             else hess_P(active_points)
    #         )

    #         for i, (point, gradient, hessian) in enumerate(
    #             zip(active_points, active_gradients, hessians)
    #         ):
    #             try:
    #                 d = np.linalg.solve(hessian, -gradient)  # Newton step
    #                 if self.projection == "sphere":
    #                     projection = np.eye(len(point)) - np.outer(point, point)
    #                     d = projection @ d
    #                 new_points[i] = point + d
    #             except np.linalg.LinAlgError:
    #                 new_points[i] = point + 0.1 * gradient
    #         projected_new_points = self.project_into_domain(new_points).copy()
    #         lsi_set[active_indices] = projected_new_points

    #         P_vals[active_indices] = p_norm(projected_new_points)
    #         new_gradients = grad_P(projected_new_points)
    #         gradients[active_indices] = new_gradients
    #         gradients_norms[active_indices] = np.linalg.norm(new_gradients, axis=1)

    #         point_steps += 1

    #     hat_ind = np.argmax(P_vals - original_P_vals)
    #     x_hat = lsi_set[hat_ind]
    #     check_ind = np.argmax(P_vals + 2 * self.R * gradients_norms)
    #     x_check = lsi_set[check_ind]
    #     max_ind = np.argmax(P_vals)
    #     max_val = P_vals[max_ind]
    #     phi_val = np.max(self.M * (max_val - self.alpha), 0) + q_u
    #     if phi_val >= self.M * epsilon:
    #         x_tilde = lsi_set[max_ind]
    #     else:
    #         x_tilde = np.array([])
    #     return x_hat, x_tilde, x_check, lsi_set

    def lsi(
        self,
        p_u: Callable,
        u: Measure,
        epsilon: float,
        Phi_A: float,
    ) -> tuple:
        # Implementation of the local support improver
        if not len(u.coefficients):
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                -1,
            )  # No points to improve around
        q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
        p_norm = lambda x: np.abs(p_u(x))
        max_P_A = np.max(p_norm(u.support))
        grad_P = self.grad_P(u)
        hess_P = self.hess_P(u)

        sorting_indices = np.argsort(p_norm(u.support))[::-1]
        full_set = u.support.copy()[sorting_indices]
        lsi_set = np.array([full_set[0]])
        for point in full_set[1:]:
            if np.linalg.norm(point - lsi_set, axis=1).min() > 2 * self.R:
                lsi_set = np.vstack((lsi_set, point))

        original_P_vals = p_norm(lsi_set)
        processing_array = np.array([True for _ in range(len(lsi_set))])
        P_vals = original_P_vals
        gradients = grad_P(lsi_set)
        gradients_norms = np.linalg.norm(gradients, axis=1)

        point_steps = 0
        while any(processing_array) and point_steps < self.stop_search:
            condition_1_lhs = np.maximum(
                P_vals - original_P_vals, self.machine_precision
            )
            condition_2_rhs = np.maximum(
                np.minimum(
                    (np.maximum(P_vals - max(self.alpha, max_P_A), 0) + Phi_A / self.M)
                    / (4 * self.R),
                    Phi_A,
                ),
                0.5 * self.machine_precision,
            )
            condition_1 = condition_1_lhs < 2 * self.R * gradients_norms
            condition_2 = gradients_norms > condition_2_rhs
            processing_array = condition_1 | condition_2  # element-wise or

            if not any(processing_array):
                break

            active_indices = np.where(processing_array)[0]
            active_points = lsi_set[active_indices]
            new_points = np.zeros(active_points.shape)
            active_gradients = gradients[active_indices]
            hessians = hess_P(active_points)

            for i, (point, gradient, hessian) in enumerate(
                zip(active_points, active_gradients, hessians)
            ):
                try:
                    d = np.linalg.solve(hessian, -gradient)  # Newton step
                    if self.projection == "sphere":
                        projection = np.eye(len(point)) - np.outer(point, point)
                        d = projection @ d
                    new_points[i] = point + d
                except np.linalg.LinAlgError:
                    new_points[i] = point + 0.1 * gradient
            projected_new_points = self.project_into_domain(new_points).copy()
            lsi_set[active_indices] = projected_new_points

            P_vals[active_indices] = p_norm(projected_new_points)
            new_gradients = grad_P(projected_new_points)
            gradients[active_indices] = new_gradients
            gradients_norms[active_indices] = np.linalg.norm(new_gradients, axis=1)

            point_steps += 1

        hat_ind = np.argmax(P_vals - original_P_vals)
        x_hat = lsi_set[hat_ind]
        check_ind = np.argmax(P_vals + 2 * self.R * gradients_norms)
        x_check = lsi_set[check_ind]
        max_ind = np.argmax(P_vals)
        max_val = P_vals[max_ind]
        phi_val = np.max(self.M * (max_val - self.alpha), 0) + q_u
        if phi_val >= self.M * epsilon:
            x_tilde = lsi_set[max_ind]
        else:
            x_tilde = np.array([])
        return x_hat, x_tilde, x_check, lsi_set, hat_ind

    def get_grid(self, u: Measure) -> np.ndarray:
        if self.projection == "sphere":
            dimensions = self.Omega.shape[0]
            sample_raw = np.random.multivariate_normal(
                mean=np.zeros(dimensions),
                cov=np.eye(dimensions),
                size=self.random_grid_size,
            )
            sample_norms = np.linalg.norm(sample_raw, axis=1)
            sample = np.divide(sample_raw, sample_norms.reshape(-1, 1))
            if len(u.coefficients):
                sample = np.vstack([sample, u.support])
            return sample
        else:
            grid = (
                np.array(
                    np.meshgrid(
                        *(
                            np.linspace(
                                bound[0], bound[1], self.global_search_resolution
                            )
                            for bound in self.Omega
                        )
                    )
                )
                .reshape(len(self.Omega), -1)
                .T
            )
            if len(u.coefficients):
                grid = np.vstack([grid, u.support])
            return grid

    def global_search(
        self, u: Measure, epsilon: float, q_u: float, p_u: Callable
    ) -> tuple:
        p_norm = lambda x: np.abs(p_u(x))
        grad_P = self.grad_P(u)
        hess_P = self.hess_P(u)
        grid = self.get_grid(u)
        grid_vals = p_norm(grid)
        max_ind = np.argmax(grid_vals)
        best_val = grid_vals[max_ind]
        best_point = grid[max_ind]
        phi_val = np.max(self.M * (best_val - self.alpha), 0) + q_u
        if phi_val >= self.M * epsilon:
            return grid[max_ind], True  # Found a desired point
        processing_array = np.array(
            [True for point in grid]
        )  # If the point is still being optimized
        point_steps = 0
        while any(processing_array) and point_steps < self.stop_search:
            active_indices = np.where(processing_array)[0]
            batching_factor = (
                len(self.target) * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
                + 2 * self.Omega.shape[0]
                + 1
            )
            batch_size = int(self.batching_constant // batching_factor)
            for batch in gen_batches(len(active_indices), batch_size):
                batch_indices = active_indices[batch]
                batch_points = grid[batch_indices]
                new_points = np.zeros(batch_points.shape)
                gradients = grad_P(batch_points)
                hessians = hess_P(batch_points)
                for i, (point, gradient, hessian) in enumerate(
                    zip(
                        batch_points,
                        gradients,
                        hessians,
                    )
                ):
                    try:
                        d = np.linalg.solve(hessian, -gradient)  # Newton step
                        if self.projection == "sphere":
                            projection = np.eye(len(point)) - np.outer(point, point)
                            d = projection @ d
                        new_points[i] = point + d
                    except np.linalg.LinAlgError:
                        new_points[i] = point + 0.1 * gradient
                projected_new_points = self.project_into_domain(new_points).copy()
                p_vals = p_norm(projected_new_points)
                max_ind = np.argmax(p_vals)
                max_val = p_vals[max_ind]
                if max_val > best_val:
                    best_val = max_val
                    best_point = projected_new_points[max_ind]
                    phi_val = np.max(self.M * (best_val - self.alpha), 0) + q_u
                    if phi_val >= self.M * epsilon:
                        return best_point, True  # Found a desired point
                grid[batch_indices] = projected_new_points
                grid_vals[batch_indices] = p_vals

                del new_points
                del projected_new_points
                del gradients
                del hessians
                del p_vals

            point_steps += 1

        return (
            best_point,
            False,
        )  # Found the global maximum, but it does not satisfy the condition

    # def build_V(self, u: Measure, xi: np.ndarray, xi_val: float, old_V: list) -> tuple:
    #     if not len(old_V):
    #         old_V = [[1000] * self.Omega.shape[0]]
    #     V_local = []
    #     mu = 0
    #     sign = np.sign(xi_val)
    #     support_distances = np.linalg.norm(u.support - xi, axis=1)
    #     for s_dist, x, c in zip(support_distances, u.support, u.coefficients):
    #         old_distances = np.linalg.norm(np.array(old_V) - x, axis=1)
    #         if (
    #             all(old_distances > self.machine_precision)
    #             and np.sign(c) == sign
    #             and s_dist < 2 * self.R
    #         ):
    #             V_local.append(x)
    #             mu += abs(c)
    #     return V_local, mu

    # def local_measure_constructor(
    #     self, p_u: Callable, u: Measure, x_hat: np.ndarray, lsi_set: list
    # ) -> Measure:
    #     new_support = [x_hat]
    #     x_hat_val = p_u(x_hat)[0]
    #     all_V, mu = self.build_V(u, x_hat, x_hat_val, [])
    #     new_coefficients = [mu * np.sign(p_u(x_hat)[0])]
    #     p_lsi = p_u(lsi_set)
    #     P_lsi = np.abs(p_lsi)
    #     sorted_indices = np.argsort(P_lsi)[::-1]
    #     lsi_set_sorted = lsi_set[sorted_indices]
    #     p_lsi_sorted = p_lsi[sorted_indices]
    #     for x, x_val in zip(lsi_set_sorted, p_lsi_sorted):
    #         V_x, mu_x = self.build_V(u, x, x_val, all_V)
    #         if mu_x:
    #             new_support.append(x)
    #             new_coefficients.append(mu * np.sign(p_u(x)[0]))
    #         all_V += V_x
    #     # Add unused old supports and coefficients
    #     if not len(all_V):
    #         all_V = [[1000] * self.Omega.shape[0]]
    #     for x, c in zip(u.support, u.coefficients):
    #         all_distances = np.linalg.norm(np.array(all_V) - x, axis=1)
    #         if all(all_distances > self.machine_precision):
    #             new_support.append(x)
    #             new_coefficients.append(c)
    #     return Measure(new_support, new_coefficients)

    def local_measure_constructor(self, u: Measure, lsi_set: np.ndarray) -> Measure:
        new_support = lsi_set.copy()
        new_coefficients = []
        for point in new_support:
            support_distances = np.linalg.norm(u.support - point, axis=1)
            local_coefs = u.coefficients[support_distances < 2 * self.R]
            new_coefficients.append(np.sum(np.abs(local_coefs)))
        to_return = Measure(new_support, new_coefficients)
        return Measure(new_support, new_coefficients)

    def finite_dimensional_step(self, u_plus: Measure, Psi: float) -> Measure:
        if not len(u_plus.coefficients):
            return u_plus
        K_support = self.kappa(u_plus.support).T
        ssn = SSN(K=K_support, alpha=self.alpha, target=self.target, M=self.M)
        u_raw = ssn.solve(tol=Psi, u_0=u_plus.coefficients)
        u_raw[np.abs(u_raw) < self.machine_precision] = 0
        # Reconstruct u
        u = Measure(
            support=u_plus.support[u_raw != 0].copy(),
            coefficients=u_raw[u_raw != 0].copy(),
        )
        return u

    def drop_step(self, u: Measure) -> tuple:
        p_u = self.p(u)
        p_vals = p_u(u.support)
        P_vals = np.abs(p_vals)
        vals_signs = np.sign(p_vals)
        drop_support = []
        drop_coefficients = []
        for point, coef, val, sign in zip(
            u.support, u.coefficients, P_vals, vals_signs
        ):
            if val >= self.alpha - 0.5 * self.sigma and np.sign(coef) == sign:
                drop_support.append(point)
                drop_coefficients.append(coef)
        if len(drop_support) != len(u.support):
            drop_u = Measure(drop_support, drop_coefficients)
            drop_j = self.j(drop_u)
            true_j = self.j(u)
            if drop_j < true_j:
                return drop_u, True
        return u, False

    def lpdap(self, tol: float, u_0: Measure = Measure(), Psi_0: float = 1) -> tuple:
        u = u_0 * 1
        p_u = self.p(u)
        q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
        u_plus = u_0 * 1
        if len(u.coefficients):
            max_P_A = np.max(np.abs(p_u(u.support)))
        else:
            max_P_A = 0
        Phi_A = max(self.M * (max_P_A - self.alpha), 0) + q_u
        epsilon = 0.5 * self.j(u) / self.M
        Psi_k = min(Psi_0, self.Psi_0)
        Phi_k = 1e8
        choice = "N/A"
        dropped = False
        dropped_tot = 0
        global_valid = False
        k = 1
        s = 1
        Phi_ks = []
        initial_time = time.time()
        times = []
        supports = []
        objective_values = []
        steps = []
        while Phi_k > tol:
            if len(u_plus.coefficients):
                # Low-dimensional step
                u_dropped, dropped = self.drop_step(u_plus)
                u = self.finite_dimensional_step(u_dropped, Psi_k)
                dropped_tot += dropped
                p_u = self.p(u)
                q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
                max_P_A = np.max(np.abs(p_u(u.support)))
                Phi_A = max(self.M * (max_P_A - self.alpha), 0) + q_u

            x_hat_lsi, x_tilde_lsi, x_check_lsi, lsi_set, hat_ind = self.lsi(
                p_u, u, epsilon, Phi_A
            )
            grad_P = self.grad_P(u)
            if len(lsi_set):
                logging.info(np.linalg.norm(grad_P(lsi_set), axis=1))
            if len(x_tilde_lsi):
                x_k = x_tilde_lsi
                global_valid = True
            else:
                x_k, global_valid = self.global_search(u, epsilon, q_u, p_u)
                logging.info(np.linalg.norm(grad_P(np.array([x_k])), axis=1))

            # Build Phi_k
            Phi_k_x_k = np.max(self.M * (np.abs(p_u(x_k))[0] - self.alpha), 0) + q_u
            if len(lsi_set):
                Phi_k_lsi = (
                    np.max(self.M * (np.max(np.abs(p_u(lsi_set))) - self.alpha), 0)
                    + q_u
                )
            else:
                Phi_k_lsi = 0
            Phi_k = max(Phi_k_x_k, Phi_k_lsi)

            # update metrics
            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            objective_values.append(self.j(u))
            Phi_ks.append(Phi_k)

            # Prepare and check for recompute
            # constant = (
            #     12
            #     * self.M
            #     * self.L
            #     * self.norm_K_star
            #     * np.sqrt(Phi_A)
            #     / np.sqrt(self.gamma)
            #     + 2 * Phi_A
            # )
            # if constant > Phi_k and Psi_k > self.machine_precision:
            if Phi_A > 0.5 * Phi_k and Psi_k > self.machine_precision:
                # Recompute step and update running metrics
                Psi_k = max(Psi_k / 2, self.machine_precision)
                logging.info(
                    f"Recompute {s}, Lazy: {global_valid}, Psi_k:{Psi_k:.3E}, Phi_k:{Phi_k:.3E}"  # , constant:{constant:.3E}, dropped:{dropped}"
                )
                steps.append("recompute")
                s += 1
                continue

            # LGCG step
            eta = min(1, Phi_k_x_k / self.C)
            if Phi_k_x_k > q_u:
                v_k = Measure([x_k], [self.M * np.sign(p_u(x_k)[0])])
            else:
                v_k = Measure()
            u_plus_hat = u * (1 - eta) + v_k * eta
            if not global_valid:
                # We have a global maximum x_k
                epsilon = 0.5 * Phi_k_x_k / self.M

            # LSI step
            if len(lsi_set):
                v_tilde = self.local_measure_constructor(u, lsi_set)
                if len(v_tilde.coefficients):
                    mu_k = v_tilde.coefficients[hat_ind]
                else:
                    mu_k = 0
                c_1k = (
                    4
                    * self.norm_kappa1
                    * np.sqrt(mu_k)
                    * (
                        1
                        + np.sqrt(mu_k * self.Omega.shape[0])
                        * self.norm_kappa1
                        * self.L
                        * (1 + 8 * max(self.alpha, self.theta * self.R**2) / self.alpha)
                        / np.sqrt(self.gamma)
                    )
                    / self.theta
                )
                c_2k = self.theta * self.norm_kappa1 * mu_k * np.sqrt(2 * self.R) / 8
                nu = min(1, mu_k / (4 * self.M * self.L * (c_1k + c_2k) ** 2))
                # nu_constant = 4 * (
                #     32
                #     * self.L
                #     * self.M**2
                #     * self.norm_K_star_L**2
                #     * (1 + self.L * self.norm_K_star / (np.sqrt(self.gamma)))
                #     / self.theta
                # )
                # nu = min(1, abs(v_tilde.coefficients[0]) / nu_constant)
                u_plus_tilde = u * (1 - nu) + v_tilde * nu
            else:
                u_plus_tilde = u * 1  # Create a new measure with the same parameters

            # Choose best step
            if self.j(u_plus_hat) < self.j(u_plus_tilde):
                u_plus = u_plus_hat * 1
                choice = f"Step: GCG, Lazy {global_valid}"
            else:
                u_plus = u_plus_tilde * 1
                choice = f"Step: LSI, Lazy {global_valid}"

            # Update running metrics
            steps.append("normal")
            logging.info(
                f"{k}: {choice}, Phi_k: {Phi_k:.3E}, epsilon: {epsilon:.3E}, support: {u.support}, coefs: {u.coefficients}, dropped:{dropped}"
            )
            logging.info("==============================================")
            k += 1
        return u, Phi_ks, times, supports, objective_values, dropped_tot

    def pdap(self, tol: float, u_0: Measure = Measure()) -> tuple:
        k = 1
        u = u_0 * 1
        p_u = self.p(u)
        q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
        x, global_valid = self.global_search(u, 1000, q_u, p_u)
        P_value = np.abs(p_u(x))
        Phi_k = np.max(self.M * (P_value - self.alpha), 0) + q_u
        P_values = [P_value]
        initial_time = time.time()
        times = [time.time() - initial_time]
        supports = [0]
        objective_values = [self.j(u)]
        while len(u.coefficients) == 0 or Phi_k > tol:
            v_k = Measure(support=np.array([x]), coefficients=[1])
            eta = min(1, Phi_k / self.C)
            u_plus = u * (1 - eta) + v_k * eta
            u = self.finite_dimensional_step(u_plus, self.machine_precision)
            p_u = self.p(u)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)

            x, global_valid = self.global_search(u, 1000, q_u, p_u)
            P_value = np.abs(p_u(x))
            Phi_k = np.max(self.M * (P_value - self.alpha), 0) + q_u
            logging.info(
                f"{k}: Phi:{Phi_k:.3E}, support: {u.support}, coefs: {u.coefficients}, x: {x}"
            )
            logging.info("==============================================")
            k += 1
            P_values.append(P_value)
            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            objective_values.append(self.j(u))
        return u, P_values, times, supports, objective_values

    def local_merging(self, u: Measure) -> tuple:
        tuples = [(x, c) for x, c in zip(u.support, u.coefficients)]
        new_tuples = []
        already_merged = []
        for ind, (x, c) in enumerate(tuples):
            if ind not in already_merged:
                c_new = c
                already_merged.append(ind)
                for ind_local, (x_local, c_local) in enumerate(tuples):
                    if ind_local not in already_merged:
                        if np.linalg.norm(x - x_local) < 2 * self.R:
                            c_new += c_local
                            already_merged.append(ind_local)
                new_tuples.append((x, c_new))
        new_points, new_coefs = tuple(zip(*new_tuples))
        new_points = np.array(list(new_points))
        new_coefs = np.array(list(new_coefs))
        return new_points, new_coefs

    def newton_step(
        self,
        points: np.ndarray,
        coefs: np.ndarray,
        beta: float,
        damped: bool,
        damping_root: float,
    ) -> tuple:
        if any(np.abs(coefs) < beta):
            # Outside of the domain of j_tilde
            return points, coefs
        grad_j_z = self.grad_j(points, coefs)
        hess_j_z = self.hess_j(points, coefs)
        if damped:
            nu = min(
                1,
                (
                    -self.m
                    + np.sqrt(
                        self.m**2
                        + 32
                        * self.m
                        * self.L_H
                        * self.bar_m**3
                        * np.linalg.norm(grad_j_z)
                    )
                )
                / (16 * self.L_H * self.bar_m**3 * np.linalg.norm(grad_j_z)),
            )
            nu = nu ** (1 / damping_root)
        else:
            nu = 1
        try:
            update_direction = np.linalg.solve(hess_j_z, -grad_j_z)
        except np.linalg.LinAlgError:
            update_direction = np.zeros_like(grad_j_z)
        if self.projection == "sphere":
            # Project the newton solution onto tangent space
            projection = np.eye(hess_j_z.shape[0])
            for i, point in enumerate(points):
                projection_part = np.eye(len(point)) - np.outer(point, point)
                projection[
                    i * len(point) : (i + 1) * len(point),
                    i * len(point) : (i + 1) * len(point),
                ] = projection_part
            update_direction = projection @ update_direction
        # Transform vector into tuples
        coefs += nu * update_direction[-len(coefs) :]
        for i, point in enumerate(points):
            new_point = point + (
                nu
                * update_direction[
                    i * self.Omega.shape[0] : (i + 1) * self.Omega.shape[0]
                ]
            )
            points[i] = new_point.copy()
        points = self.project_into_domain(points)
        return points, coefs

    def lgcg_step(self, p_u: Callable, u: Measure, epsilon: float, q_u: float) -> tuple:
        x_k, global_valid = self.global_search(u, epsilon, q_u, p_u)
        Phi = np.max(self.M * (np.abs(p_u(x_k))[0] - self.alpha), 0) + q_u
        eta = min(1, Phi / self.C)
        if Phi > q_u:
            v = Measure([x_k], [self.M * np.sign(p_u(x_k)[0])])
        else:
            v = Measure()
        u_plus = u * (1 - eta) + v * eta
        if not global_valid:
            # We have a global maximum x_k
            epsilon = 0.5 * Phi / self.M
        return u_plus, epsilon, global_valid

    def get_regularity_inequalities(
        self, points, coefs, points_plus, coefs_plus, epsilon_plus, tol
    ) -> tuple:
        grad_j_z = self.grad_j(points, coefs)
        norm_grad = np.linalg.norm(grad_j_z)
        hess_j_z = self.hess_j(points, coefs)
        j_tilde_diff = self.j_tilde(points_plus, coefs_plus) - self.j_tilde(
            points, coefs
        )
        ineq73 = j_tilde_diff <= -0.125 * self.m * norm_grad**2
        # logging.info(f"diff: {j_tilde_diff}, grad_norm: {norm_grad}")
        try:
            ineq75 = (
                np.linalg.norm(np.linalg.solve(hess_j_z, grad_j_z)) / norm_grad
                <= 2 * self.bar_m
            )
        except np.linalg.LinAlgError:
            ineq75 = False
        if self.M * epsilon_plus <= self.C:
            ineq78 = norm_grad**2 >= self.M**2 * epsilon_plus**2 / (
                2 * self.bar_m * self.C
            )
        else:
            ineq78 = norm_grad**2 >= (2 * self.M * epsilon_plus - self.C) / (
                2 * self.bar_m
            )
        tol_ineq = 2 * self.M * epsilon_plus >= tol
        M_ineq = np.linalg.norm(coefs_plus, ord=1) <= self.M
        return ineq73, ineq75, ineq78, tol_ineq, M_ineq

    def newton(
        self,
        tol: float,
        lgcg_frequency: int = 5,
        beta: float = 1e-5,
        damped: bool = False,
        damping_root: float = 1,
        u_0: Measure = Measure(),
        Psi_0: float = 1,
    ) -> tuple:
        u_minus = u_0 * 1
        epsilon = 0.5 * self.j(u_minus) / self.M
        Psi_k = min(Psi_0, self.Psi_0)
        k = 1
        dropped = False
        dropped_tot = 0
        initial_time = time.time()
        times = [time.time() - initial_time]
        supports = [0]
        inner_loop = [0]
        objective_values = [self.j(u_minus)]
        inner_lazy = 0
        inner_total = 0
        outer_lazy = 0
        outer_total = 0
        while 2 * self.M * epsilon > tol:
            # LocalRoutine
            if len(u_minus.coefficients):
                s = 1
                points, coefs = self.local_merging(u_minus)
                u_ks = Measure(points, coefs)
                p_u_ks = self.p(u_ks)
                q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
                epsilon_ks = epsilon + 0.5 * (self.j(u_ks) - self.j(u_minus)) / self.M
                points_plus, coefs_plus = self.newton_step(
                    points.copy(), coefs.copy(), beta, damped, damping_root
                )
                u_ks_plus = Measure(points_plus, coefs_plus)
                u_hat_plus, epsilon_ks_plus, global_valid = self.lgcg_step(
                    p_u_ks, u_ks, epsilon_ks, q_u_ks
                )
                inner_lazy += int(global_valid)
                inner_total += 1

                # Regularity conditions
                ineq73, ineq75, ineq78, tol_ineq, M_ineq = (
                    self.get_regularity_inequalities(
                        points, coefs, points_plus, coefs_plus, epsilon_ks_plus, tol
                    )
                )

                logging.info(f"{ineq73}, {ineq75}, {ineq78}, {tol_ineq}, {M_ineq}")
                while ineq73 and ineq75 and ineq78 and tol_ineq and M_ineq:
                    times.append(time.time() - initial_time)
                    supports.append(len(u_ks.support))
                    inner_loop.append(1)
                    objective_values.append(self.j(u_ks))
                    logging.info(
                        f"{k}, {s}: lazy: {global_valid}, support: {u_ks.support}, coefs: {u_ks.coefficients}, epsilon: {epsilon_ks_plus}, objective: {self.j(u_ks):.3E}"
                    )

                    s += 1
                    points, coefs = points_plus.copy(), coefs_plus.copy()
                    u_ks = u_ks_plus * 1
                    p_u_ks = self.p(u_ks)
                    q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
                    points_plus, coefs_plus = self.newton_step(
                        points.copy(), coefs.copy(), beta, damped, damping_root
                    )
                    u_ks_plus = Measure(points_plus, coefs_plus)
                    if s % lgcg_frequency == 0:
                        epsilon_ks = epsilon_ks_plus
                        u_hat_plus, epsilon_ks_plus, global_valid = self.lgcg_step(
                            p_u_ks, u_ks, epsilon_ks, q_u_ks
                        )
                        inner_lazy += int(global_valid)
                        inner_total += 1
                    else:
                        global_valid = "N/A"
                    ineq73, ineq75, ineq78, tol_ineq, M_ineq = (
                        self.get_regularity_inequalities(
                            points, coefs, points_plus, coefs_plus, epsilon_ks_plus, tol
                        )
                    )
                    logging.info(f"{ineq73}, {ineq75}, {ineq78}, {tol_ineq}, {M_ineq}")

                # Choose best iterate so far
                all_iterates = [u_minus, u_ks, u_ks_plus, u_hat_plus]
                iterate_values = [self.j(iterate) for iterate in all_iterates]
                choice_index = np.argmin(iterate_values)
                u_plus = all_iterates[choice_index] * 1
                if not tol_ineq:
                    epsilon = epsilon_ks_plus
            else:
                choice_index = 0
                u_plus = u_minus * 1

            if len(u_plus.coefficients):
                u_dropped, dropped = self.drop_step(u_plus)
                u = self.finite_dimensional_step(u_dropped, Psi_k)
                dropped_tot += dropped
            else:
                u = u_plus * 1
            p_u = self.p(u)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
            Psi_k = max(Psi_k / 2, self.machine_precision)

            u_minus, epsilon, global_valid = self.lgcg_step(p_u, u, epsilon, q_u)
            outer_lazy += int(global_valid)
            outer_total += 1

            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            inner_loop.append(0)
            objective_values.append(self.j(u))
            logging.info(
                f"{k}: choice: {choice_index}, lazy: {global_valid}, support: {u.support}, epsilon: {epsilon}, objective: {self.j(u):.3E}, dropped:{dropped}"
            )
            k += 1

        return (
            u,
            times,
            supports,
            inner_loop,
            inner_lazy,
            inner_total,
            outer_lazy,
            outer_total,
            objective_values,
            dropped_tot,
        )
