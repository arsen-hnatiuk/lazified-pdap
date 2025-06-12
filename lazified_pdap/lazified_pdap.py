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
                -1,
            )  # No points to improve around
        q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
        p_norm = lambda x: np.abs(p_u(x))
        grad_P = self.grad_P(u)
        hess_P = self.hess_P(u)

        sorting_indices = np.argsort(p_norm(u.support))[::-1]
        full_set = u.support.copy()[sorting_indices]
        lsi_set = np.array([full_set[0]])
        for point in full_set[1:]:
            if np.linalg.norm(point - lsi_set, axis=1).min() > 2 * self.R:
                lsi_set = np.vstack((lsi_set, point))
        start_set = lsi_set.copy()

        original_P_vals = p_norm(lsi_set)
        P_vals = original_P_vals
        gradients = grad_P(lsi_set)
        gradients_norms = np.linalg.norm(gradients, axis=1)

        condition_1_lhs = np.maximum(P_vals - original_P_vals, self.machine_precision)
        condition_1 = condition_1_lhs < 2 * self.R * gradients_norms
        condition_2 = P_vals <= self.alpha - self.sigma / 2
        condition_3 = gradients_norms > Phi_A
        processing_array = condition_1 | condition_2 | condition_3  # element-wise or

        point_steps = 0
        while any(processing_array) and point_steps < self.stop_search:
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

            condition_1_lhs = np.maximum(
                P_vals - original_P_vals, self.machine_precision
            )
            condition_1 = condition_1_lhs < 2 * self.R * gradients_norms
            condition_2 = P_vals <= self.alpha - self.sigma / 2
            condition_3 = gradients_norms > Phi_A
            distances = np.linalg.norm(lsi_set - start_set, axis=1)
            distance_condition = distances > 2 * self.R
            processing_array = (
                condition_1 | condition_2 | distance_condition | condition_3
            )  # element-wise or

            point_steps += 1

        # discard invalid points
        lsi_set = lsi_set[~processing_array]
        P_vals = P_vals[~processing_array]
        original_P_vals = original_P_vals[~processing_array]

        if not len(lsi_set):
            return (
                np.array([]),
                np.array([]),
                -1,
            )

        hat_ind = np.argmax(P_vals - original_P_vals)
        max_ind = np.argmax(P_vals)
        max_val = P_vals[max_ind]
        phi_val = self.M * max((max_val - self.alpha), 0) + q_u
        if phi_val >= self.M * epsilon:
            x_tilde = lsi_set[max_ind]
        else:
            x_tilde = np.array([])
        return x_tilde, lsi_set, hat_ind

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
        point_steps = 0
        while point_steps < self.stop_search:
            batching_factor = (
                len(self.target) * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
                + 2 * self.Omega.shape[0]
                + 1
            )
            batch_size = int(self.batching_constant // batching_factor)
            for batch in gen_batches(len(grid), batch_size):
                batch_points = grid[batch]
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
                grid[batch] = projected_new_points
                grid_vals[batch] = p_vals

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

    def local_measure_constructor(self, u: Measure, lsi_set: np.ndarray) -> Measure:
        new_support = lsi_set.copy()
        new_coefficients = []
        for point in new_support:
            support_distances = np.linalg.norm(u.support - point, axis=1)
            local_coefs = u.coefficients[support_distances < 2 * self.R]
            new_coefficients.append(np.sum(local_coefs))
        to_return = Measure(new_support, new_coefficients)
        return to_return

    def finite_dimensional_step(
        self, u: Measure, Psi: float, mode: str = "unconstrained"
    ) -> tuple:
        if not len(u.coefficients):
            return u
        K_support = self.kappa(u.support).T
        if mode == "positive":
            signs = np.sign(u.coefficients)
            K_support = np.multiply(K_support, signs)
            u_0 = np.abs(u.coefficients)
            # u_0 = np.clip(u.coefficients * signs, 0, np.max(np.abs(u.coefficients)))
        else:
            u_0 = u.coefficients.copy()
        ssn = SSN(
            K=K_support, alpha=self.alpha, target=self.target, M=self.M, mode=mode
        )
        u_raw = ssn.solve(tol=Psi, u_0=u_0)
        raw_Psi = ssn.Psi(u_raw)
        # u_raw[np.abs(u_raw) < self.machine_precision] = 0
        if mode == "positive":
            u_raw = u_raw * signs
        # Reconstruct u
        u_plus = Measure(
            support=u.support[u_raw != 0].copy(),
            coefficients=u_raw[u_raw != 0].copy(),
        )
        return u_plus, raw_Psi
        # if mode == "positive":
        # if self.j(u_plus) <= self.j(u):  # < self.machine_precision:
        # return u_plus, raw_Psi
        # else:
        #     return u, ssn.Psi(u_0)
        return u_plus, raw_Psi  # Unconstrained case

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
        Phi_A = self.M * max((max_P_A - self.alpha), 0) + q_u
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
        epsilons = []
        initial_time = time.time()
        times = []
        supports = []
        objective_values = []
        steps = []
        while Phi_k > tol:
            if len(u_plus.coefficients):
                # Low-dimensional step
                u_dropped, dropped = self.drop_step(u_plus)
                u, finite_Phi = self.finite_dimensional_step(
                    u_dropped, Psi_k, mode="positive"
                )
                dropped_tot += dropped
                p_u = self.p(u)
                q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
                max_P_A = np.max(np.abs(p_u(u.support)))
                Phi_A = self.M * max((max_P_A - self.alpha), 0) + q_u

            x_tilde_lsi, lsi_set, hat_ind = self.lsi(p_u, u, epsilon, Phi_A)
            if len(x_tilde_lsi):
                x_k = x_tilde_lsi
                global_valid = True
            else:
                x_k, global_valid = self.global_search(u, epsilon, q_u, p_u)

            # Build Phi_k
            Phi_k = self.M * max((np.abs(p_u(x_k))[0] - self.alpha), 0) + q_u

            # update metrics
            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            objective_values.append(self.j(u))
            Phi_ks.append(Phi_k)
            epsilons.append(epsilon)

            # Check for recompute
            if (
                len(u_plus.coefficients)
                and finite_Phi > 0.5 * Phi_k
                and Psi_k > self.machine_precision
            ):
                # Recompute step and update running metrics
                Psi_k = max(Psi_k / 2, self.machine_precision)
                logging.info(
                    f"Recompute {s}, Lazy: {global_valid}, Psi_k:{Psi_k:.3E}, Phi_k:{Phi_k:.3E}, dropped:{dropped}"
                )
                steps.append("recompute")
                s += 1
                continue

            # LGCG step
            eta = max(min(1, Phi_k / self.C), self.machine_precision)
            if Phi_k > q_u:
                v_k = Measure([x_k], [self.M * np.sign(p_u(x_k)[0])])
            else:
                v_k = Measure()
            u_plus_hat = u * (1 - eta) + v_k * eta
            if not global_valid:
                # We have a global maximum x_k
                epsilon = 0.5 * Phi_k / self.M

            # LSI step
            if len(lsi_set):
                v_tilde = self.local_measure_constructor(u, lsi_set)
                if len(v_tilde.coefficients):
                    mu_k = v_tilde.coefficients[hat_ind]
                else:
                    mu_k = 0
                denominator = (
                    16
                    * self.M
                    * self.L
                    * self.norm_kappa1**2
                    * (
                        2 * self.M * np.sqrt(self.R / self.theta)
                        + 2
                        * self.M
                        * self.norm_kappa1**2
                        * self.L
                        / (self.theta * np.sqrt(self.gamma))
                        + np.sqrt(self.M / self.theta)
                    )
                    ** 2
                )
                nu = max(min(1, mu_k / denominator), self.machine_precision)
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
                f"{k}: {choice}, Phi_k: {Phi_k:.3E}, epsilon: {epsilon:.3E}, support: {len(u.support)}, dropped:{dropped}"
            )
            logging.info("==============================================")
            k += 1
        return u, Phi_ks, times, supports, objective_values, dropped_tot, epsilons

    def pdap(self, tol: float, u_0: Measure = Measure()) -> tuple:
        k = 1
        u = u_0 * 1
        p_u = self.p(u)
        q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
        x, global_valid = self.global_search(u, 1000, q_u, p_u)
        P_value = np.abs(p_u(x))[0]
        Phi_k = self.M * max((P_value - self.alpha), 0) + q_u
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
            P_value = np.abs(p_u(x))[0]
            Phi_k = self.M * max((P_value - self.alpha), 0) + q_u
            logging.info(f"{k}: Phi:{Phi_k:.3E}, support: {len(u.support)}")
            logging.info("==============================================")
            k += 1
            P_values.append(P_value)
            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            objective_values.append(self.j(u))
        return u, P_values, times, supports, objective_values

    def local_merging(self, u: Measure) -> tuple:
        p_u = self.p(u)
        p_norm = lambda x: np.abs(p_u(x))
        sorting_indices = np.argsort(p_norm(u.support))[::-1]
        full_set = u.support.copy()[sorting_indices]
        merge_set = np.array([full_set[0]])
        merge_coefs = [
            np.sum(
                u.coefficients[
                    np.linalg.norm(u.support - full_set[0], axis=1) < 2 * self.R
                ]
            )
        ]
        for point in full_set[1:]:
            if np.linalg.norm(point - merge_set, axis=1).min() > 2 * self.R:
                merge_set = np.vstack((merge_set, point))
                merge_coefs.append(
                    np.sum(
                        u.coefficients[
                            np.linalg.norm(u.support - point, axis=1) < 2 * self.R
                        ]
                    )
                )
        return merge_set, np.array(merge_coefs)

        # tuples = [(x, c) for x, c in zip(u.support, u.coefficients)]
        # new_tuples = []
        # already_merged = []
        # for ind, (x, c) in enumerate(tuples):
        #     if ind not in already_merged:
        #         c_new = c
        #         already_merged.append(ind)
        #         for ind_local, (x_local, c_local) in enumerate(tuples):
        #             if ind_local not in already_merged:
        #                 if np.linalg.norm(x - x_local) < 2 * self.R:
        #                     c_new += c_local
        #                     already_merged.append(ind_local)
        #         new_tuples.append((x, c_new))
        # new_points, new_coefs = tuple(zip(*new_tuples))
        # new_points = np.array(list(new_points))
        # new_coefs = np.array(list(new_coefs))
        # return new_points, new_coefs

    def newton_step(
        self,
        points: np.ndarray,
        coefs: np.ndarray,
        damped: bool,
        damping_root: float,
    ) -> tuple:
        grad_j_z = self.grad_j(points, coefs)
        hess_j_z = self.hess_j(points, coefs)
        points_new = np.empty_like(points)
        coefs_new = np.empty_like(coefs)
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
        coafs_new = coefs + nu * update_direction[-len(coefs) :]
        for i, point in enumerate(points):
            new_point = point + (
                nu
                * update_direction[
                    i * self.Omega.shape[0] : (i + 1) * self.Omega.shape[0]
                ]
            )
            points_new[i] = new_point.copy()
        points_new = self.project_into_domain(points_new)
        return points_new, coefs_new

    def lgcg_step(self, p_u: Callable, u: Measure, epsilon: float, q_u: float) -> tuple:
        x_k, global_valid = self.global_search(u, epsilon, q_u, p_u)
        Phi = self.M * max((np.abs(p_u(x_k))[0] - self.alpha), 0) + q_u
        eta = max(min(1, Phi / self.C), self.machine_precision)
        if Phi > q_u:
            v = Measure([x_k], [self.M * np.sign(p_u(x_k)[0])])
        else:
            v = Measure()
        u_plus = u * (1 - eta) + v * eta
        if not global_valid:
            # We have a global maximum x_k
            epsilon = 0.5 * Phi / self.M
        return u_plus, epsilon, global_valid

    # def get_regularity_inequalities(
    #     self, points, coefs, points_plus, coefs_plus, epsilon_plus, tol
    # ) -> tuple:
    #     grad_j_z = self.grad_j(points, coefs)
    #     norm_grad = np.linalg.norm(grad_j_z)
    #     hess_j_z = self.hess_j(points, coefs)
    #     j_tilde_diff = self.j_tilde(points_plus, coefs_plus) - self.j_tilde(
    #         points, coefs
    #     )
    #     ineq73 = j_tilde_diff <= -0.125 * self.m * norm_grad**2
    #     try:
    #         ineq75 = (
    #             np.linalg.norm(np.linalg.solve(hess_j_z, grad_j_z)) / norm_grad
    #             <= 2 * self.bar_m
    #         )
    #     except np.linalg.LinAlgError:
    #         ineq75 = False
    #     if self.M * epsilon_plus <= self.C:
    #         ineq78 = norm_grad**2 >= self.M**2 * epsilon_plus**2 / (
    #             2 * self.bar_m * self.C
    #         )
    #     else:
    #         ineq78 = norm_grad**2 >= (2 * self.M * epsilon_plus - self.C) / (
    #             2 * self.bar_m
    #         )
    #     tol_ineq = 2 * self.M * epsilon_plus >= tol
    #     M_ineq = np.linalg.norm(coefs_plus, ord=1) <= self.M
    #     return ineq73, ineq75, ineq78, tol_ineq, M_ineq

    def first_inequalities(
        self,
        points: np.ndarray,
        coefs: np.ndarray,
        points_new: np.ndarray,
        coefs_new: np.ndarray,
    ) -> list:
        output_bools = []

        # Domain test
        projected_points_new = self.project_into_domain(points_new)
        projection_distance = np.linalg.norm(points_new - projected_points_new)
        output_bools.append(projection_distance == 0)

        # M test
        output_bools.append(np.linalg.norm(coefs_new, ord=1) <= self.M)

        # Descent test
        grad_j_z = self.grad_j(points, coefs)
        norm_grad = np.linalg.norm(grad_j_z)
        j_tilde_diff = self.j_tilde(points_new, coefs_new) - self.j_tilde(points, coefs)
        output_bools.append(j_tilde_diff <= -0.125 * self.m * norm_grad**2)

        return output_bools

    def second_inequality(
        self, points: np.ndarray, coefs: np.ndarray, epsilon: float
    ) -> bool:
        grad_j_z = self.grad_j(points, coefs)
        norm_grad = np.linalg.norm(grad_j_z)
        if self.M * epsilon <= self.C:
            ineq = norm_grad**2 >= self.M**2 * epsilon**2 / (2 * self.bar_m * self.C)
        else:
            ineq = norm_grad**2 >= (2 * self.M * epsilon - self.C) / (2 * self.bar_m)

    def newton(
        self,
        tol: float,
        drop_frequency: int = 5,
        damped: bool = False,
        damping_root: float = 1,
        u_0: Measure = Measure(),
        Psi_0: float = 1,
    ) -> tuple:
        u = u_0 * 1
        epsilon = 0.5 * self.j(u) / self.M
        Psi_k = min(Psi_0, self.Psi_0)
        k = 1
        dropped = False
        dropped_tot = 0
        initial_time = time.time()
        times = [time.time() - initial_time]
        supports = [0]
        inner_loop = [0]
        objective_values = [self.j(u)]
        epsilons = [epsilon]
        lgcg_lazy = 0
        lgcg_total = 0
        while 2 * self.M * epsilon > tol:
            p_u = self.p(u)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
            u_gcg, epsilon, global_valid = self.lgcg_step(p_u, u, epsilon, q_u)
            lgcg_lazy += int(global_valid)
            lgcg_total += 1

            if len(u_gcg.coefficients):
                u_drop, dropped = self.drop_step(u_gcg)
                u, finite_psi = self.finite_dimensional_step(u_drop, Psi_k)
                dropped_tot += dropped
            else:
                u = u_gcg * 1

            points, coefs = self.local_merging(u)
            u_ks = Measure(points, coefs)
            epsilon_ks = epsilon + 0.5 * (self.j(u_ks) - self.j(u_drop)) / self.M
            p_u_ks = self.p(u_ks)
            q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
            u_ks_gcg = u_ks * 1
            u_ks_new = u_ks * 1
            u_lm = u_ks * 1

            while len(u_ks.coefficients):
                # Inner loop
                s = 1
                points_new, coefs_new = self.newton_step(
                    points, coefs, damped, damping_root
                )
                u_ks_new = Measure(points_new, coefs_new)

                first_inequalities = self.first_inequalities(
                    points, coefs, points_new, coefs_new
                )
                if not all(first_inequalities):
                    logging.info(first_inequalities)
                    break
                second_inequality = self.second_inequality(points, coefs, epsilon_ks)
                if not second_inequality:
                    u_ks_gcg, epsilon_ks, global_valid = self.lgcg_step(
                        p_u_ks, u_ks, epsilon_ks, q_u_ks
                    )
                    lgcg_lazy += int(global_valid)
                    lgcg_total += 1
                else:
                    global_valid = "N/A"
                second_inequality = self.second_inequality(points, coefs, epsilon_ks)
                if not second_inequality:
                    break

                if s % drop_frequency == 0:
                    u_ks_drop, dropped = self.drop_step(u_ks_new)
                    dropped_tot += dropped
                    points, coefs = self.local_merging(u_ks_drop)
                    u_ks = Measure(points, coefs)
                    epsilon_ks = (
                        epsilon_ks + 0.5 * (self.j(u_ks) - self.j(u_ks_drop)) / self.M
                    )
                else:
                    u_ks = u_ks_new * 1
                    points = points_new.copy()
                    coefs = coefs_new.copy()
                p_u_ks = self.p(u_ks)
                q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)

                # points, coefs = self.local_merging(u_minus)
                # u_ks = Measure(points, coefs)
                # p_u_ks = self.p(u_ks)
                # q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
                # epsilon_ks = epsilon + 0.5 * (self.j(u_ks) - self.j(u_minus)) / self.M
                # points_plus, coefs_plus = self.newton_step(
                #     points, coefs, damped, damping_root
                # )
                # u_ks_plus = Measure(points_plus, coefs_plus)
                # u_hat_plus, epsilon_ks_plus, global_valid = self.lgcg_step(
                #     p_u_ks, u_ks, epsilon_ks, q_u_ks
                # )

                # inner_lazy += int(global_valid)
                # inner_total += 1
                times.append(time.time() - initial_time)
                supports.append(len(u_ks.support))
                inner_loop.append(1)
                objective_values.append(self.j(u_ks))
                epsilons.append(epsilon_ks)
                logging.info(
                    f"{k}, {s}: lazy: {global_valid}, support: {u_ks.support}, coefs: {u_ks.coefficients}, epsilon: {epsilon_ks}, objective: {self.j(u_ks):.3E}"
                )

            all_iterates = [u, u_lm, u_ks, u_ks_new, u_ks_gcg]
            iterate_values = [self.j(iterate) for iterate in all_iterates]
            choice_index = np.argmin(iterate_values)
            u = all_iterates[choice_index] * 1

            # if not tol_ineq:
            #     epsilon = epsilon_ks_plus

            # # Regularity conditions
            # ineq73, ineq75, ineq78, tol_ineq, M_ineq = (
            #     self.get_regularity_inequalities(
            #         points, coefs, points_plus, coefs_plus, epsilon_ks_plus, tol
            #     )
            # )

            # logging.info(f"{ineq73}, {ineq75}, {ineq78}, {tol_ineq}, {M_ineq}")
            # while ineq73 and ineq75 and ineq78 and tol_ineq and M_ineq:
            #     s += 1
            #     points, coefs = points_plus.copy(), coefs_plus.copy()
            #     u_ks = u_ks_plus * 1
            #     p_u_ks = self.p(u_ks)
            #     q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
            #     points_plus, coefs_plus = self.newton_step(
            #         points, coefs, damped, damping_root
            #     )
            #     u_ks_plus = Measure(points_plus, coefs_plus)
            #     if s % lgcg_frequency == 0:
            #         epsilon_ks = epsilon_ks_plus
            #         u_hat_plus, epsilon_ks_plus, global_valid = self.lgcg_step(
            #             p_u_ks, u_ks, epsilon_ks, q_u_ks
            #         )
            #         inner_lazy += int(global_valid)
            #         inner_total += 1
            #     else:
            #         global_valid = "N/A"
            #     ineq73, ineq75, ineq78, tol_ineq, M_ineq = (
            #         self.get_regularity_inequalities(
            #             points, coefs, points_plus, coefs_plus, epsilon_ks_plus, tol
            #         )
            #     )
            #     times.append(time.time() - initial_time)
            #     supports.append(len(u_ks.support))
            #     inner_loop.append(1)
            #     objective_values.append(self.j(u_ks))
            #     epsilons.append(epsilon_ks)
            #     logging.info(
            #         f"{k}, {s}: lazy: {global_valid}, support: {u_ks.support}, coefs: {u_ks.coefficients}, epsilon: {epsilon_ks_plus}, objective: {self.j(u_ks):.3E}"
            #     )
            #     logging.info(f"{ineq73}, {ineq75}, {ineq78}, {tol_ineq}, {M_ineq}")

            # # Choose best iterate so far
            # all_iterates = [u_minus, u_ks, u_ks_plus, u_hat_plus]
            # iterate_values = [self.j(iterate) for iterate in all_iterates]
            # choice_index = np.argmin(iterate_values)
            # u_plus = all_iterates[choice_index] * 1
            # if not tol_ineq:
            #     epsilon = epsilon_ks_plus
            # else:
            #     choice_index = 0
            #     u_plus = u_minus * 1

            # if len(u_plus.coefficients):
            #     u_dropped, dropped = self.drop_step(u_plus)
            #     u = self.finite_dimensional_step(u_dropped, Psi_k)
            #     dropped_tot += dropped
            # else:
            #     u = u_plus * 1
            p_u = self.p(u)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
            Psi_k = max(Psi_k / 2, self.machine_precision)

            # u_minus, epsilon, global_valid = self.lgcg_step(p_u, u, epsilon, q_u)
            # outer_lazy += int(global_valid)
            # outer_total += 1

            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            inner_loop.append(0)
            objective_values.append(self.j(u))
            epsilons.append(epsilon)
            logging.info(
                f"{k}: choice: {choice_index}, lazy: {global_valid}, support: {u.support}, epsilon: {epsilon}, objective: {self.j(u):.3E}, dropped:{dropped}"
            )
            k += 1

        return (
            u,
            times,
            supports,
            inner_loop,
            lgcg_lazy,
            lgcg_total,
            objective_values,
            dropped_tot,
            epsilons,
        )
