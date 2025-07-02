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


class NLGCGParameterFree:
    def __init__(
        self,
        target: np.ndarray,
        kappa: Callable,
        g: Callable,
        f: Callable,
        p: Callable,
        grad_P: Callable,
        hess_P: Callable,
        grad_j: Callable,
        hess_j: Callable,
        alpha: float,
        Omega: np.ndarray,
        global_search_resolution: int,
        M: float = 1e6,
        C_0: float = 1,
        dual_variable_goodness: float = 0.5,
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
        self.Omega = Omega  # Example [[0,1],[1,2]] for [0,1]x[1,2]
        self.max_radius = 1
        self.j = lambda u: self.f(u) + self.g(u.coefficients)
        self.j_tilde = lambda pos, coef: self.f(Measure(pos, coef)) + self.g(coef)
        self.u_0 = Measure()
        self.M_0 = min(
            M, self.j(self.u_0) / self.alpha
        )  # Bound on the norm of iterates
        self.M = self.M_0
        self.C_0 = C_0
        self.C_raw = self.C_0  # curvature constant without the M part
        self.global_search_resolution = global_search_resolution
        self.grad_j = grad_j
        self.hess_j = hess_j
        self.Psi_0 = 1e-3
        self.machine_precision = 1e-12
        self.stop_search = 5
        self.batching_constant = 2e8
        self.random_grid_size = random_grid_size
        self.dual_variable_goodness = dual_variable_goodness

    def project_into_domain(self, x: np.ndarray) -> np.ndarray:
        # Project an array into domain, parallelized
        for i, bounds in zip(range(x.shape[1]), self.Omega):
            column = x[:, i].copy()
            x[:, i] = np.clip(column, bounds[0], bounds[1])
        return x

    def get_grid(self, u: Measure) -> np.ndarray:
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
        if len(u.coefficients):
            grid = np.vstack([grid, u.support])
        return grid

    def global_search(
        self, u: Measure, epsilon: float, q_u: float, p_u: Callable, radius: float
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
        point_steps = 0
        while point_steps < self.stop_search and phi_val < self.M * epsilon:
            batching_factor = (
                len(self.target) * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
                + 2 * self.Omega.shape[0]
                + 1
            )
            batch_size = int(self.batching_constant // batching_factor)
            for batch in gen_batches(len(grid), batch_size):
                if phi_val >= self.M * epsilon:
                    break
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
                        new_points[i] = point + d
                    except np.linalg.LinAlgError:
                        new_points[i] = point + 0.1 * gradient
                projected_new_points = self.project_into_domain(new_points).copy()
                p_vals = p_norm(projected_new_points)
                grid[batch] = projected_new_points
                grid_vals[batch] = p_vals
                max_ind = np.argmax(p_vals)
                max_val = p_vals[max_ind]
                if max_val > best_val:
                    best_val = max_val
                    best_point = projected_new_points[max_ind]
                    phi_val = np.max(self.M * (best_val - self.alpha), 0) + q_u
                    if phi_val >= self.M * epsilon:
                        break

                del new_points
                del projected_new_points
                del gradients
                del hessians
                del p_vals

            point_steps += 1

        if phi_val >= self.M * epsilon:
            success = True  # Found a desired point
        else:
            success = False
        valid_indices = np.where(
            np.logical_and(
                grid_vals >= self.alpha,
                grid_vals > best_val * self.dual_variable_goodness,
            )
        )[0]
        valid_grid = grid[valid_indices]
        valid_grid_vals = grid_vals[valid_indices]
        order_indices = np.argsort(valid_grid_vals)[::-1]
        order_grid = valid_grid[order_indices][:100]

        if not len(order_grid):
            found_points = np.array([best_point])
        else:
            found_points = np.array([order_grid[0]])
        for point in order_grid[1:]:
            local_distances = np.linalg.norm(found_points - point, axis=1)
            if np.all(local_distances > 2 * radius):
                found_points = np.vstack((found_points, point))
        found_values = p_u(found_points)

        return (
            best_point,
            found_points,
            found_values,
            success,
        )

    def finite_dimensional_step(
        self, u: Measure, Psi: float, mode: str = "unconstrained"
    ) -> tuple:
        if not len(u.coefficients):
            return u, Psi * 2
        K_support = self.kappa(u.support).T
        if mode == "positive":
            signs = np.sign(u.coefficients)
            K_support = np.multiply(K_support, signs)
            u_0 = np.abs(u.coefficients)
        else:
            u_0 = u.coefficients.copy()
        ssn = SSN(
            K=K_support, alpha=self.alpha, target=self.target, M=self.M, mode=mode
        )
        u_raw = ssn.solve(tol=Psi, u_0=u_0)
        raw_Psi = ssn.Psi(u_raw)
        if mode == "positive":
            u_raw = u_raw * signs
        # Reconstruct u
        u_plus = Measure(
            support=u.support[u_raw != 0].copy(),
            coefficients=u_raw[u_raw != 0].copy(),
        )
        return u_plus, raw_Psi

    def drop_step(self, u: Measure) -> tuple:
        true_j = self.j(u)
        p_u = self.p(u)
        p_vals = p_u(u.support)
        vals_signs = np.sign(p_vals)
        measure_signs = np.sign(u.coefficients)
        keep_indices = np.where(vals_signs == measure_signs)[0]
        P_vals_unsorted = np.abs(p_vals[keep_indices])
        sorting_indices = np.argsort(P_vals_unsorted)[::-1]
        reduced_support = u.support[keep_indices][sorting_indices]
        reduced_coefficients = u.coefficients[keep_indices][sorting_indices]

        new_support = []
        new_coefficients = []
        for point, coef in zip(reduced_support, reduced_coefficients):
            new_support.append(point)
            new_coefficients.append(coef)
            tentative_u = Measure(new_support, new_coefficients)
            tentative_j = self.j(tentative_u)
            if tentative_j <= true_j:
                if len(new_support) == len(u.support):
                    return tentative_u, False
                else:
                    return tentative_u, True
        return u, False

    def local_merging_update_radii(self, u: Measure) -> tuple:
        if not len(u.coefficients):
            return np.array([]), np.array([]), u, []
        radii = self.compute_radii(u)
        p_u = self.p(u)
        p_norm = lambda x: np.abs(p_u(x))
        sorting_indices = np.argsort(p_norm(u.support))[::-1]
        full_set = u.support.copy()[sorting_indices]
        full_coefs = u.coefficients.copy()[sorting_indices]
        full_radii = np.array(radii)[sorting_indices]
        merged = np.array([False] * len(full_set), dtype=bool)
        cluster_points = []
        cluster_coefs = []
        for i, point in enumerate(full_set):
            if not merged[i]:
                local_distances = np.linalg.norm(full_set[~merged] - point, axis=1)
                cluster_indices = local_distances <= 2 * np.maximum(
                    full_radii[~merged], full_radii[i]
                )
                cluster_points.append(point)
                cluster_coefs.append(np.sum(full_coefs[~merged][cluster_indices]))
                merged[~merged] |= cluster_indices
        u_plus = Measure(cluster_points, cluster_coefs)
        if not len(u.support) == len(u_plus.support):
            radii = self.compute_radii(u_plus)
        return np.array(cluster_points), np.array(cluster_coefs), u_plus, radii

    def vectro_to_tuples(
        self, points: np.ndarray, coefs: np.ndarray, direction: np.ndarray
    ) -> tuple:
        points_new = points + direction[: -len(coefs)].reshape(points.shape)
        coefs_new = coefs + direction[-len(coefs) :]
        # for i, point in enumerate(points):
        #     new_point = point + (
        #         direction[i * self.Omega.shape[0] : (i + 1) * self.Omega.shape[0]]
        #     )
        #     points_new[i] = new_point.copy()
        return points_new, coefs_new

    def armijo(
        self,
        points: np.ndarray,
        coefs: np.ndarray,
        direction: np.ndarray,
        gradient: np.ndarray,
    ) -> tuple:
        beta = 0.5
        armijo_constant = 1e-4
        j_tilde_init = self.j_tilde(points, coefs)
        desired_descent = armijo_constant * gradient @ direction
        sigma = 1
        points_new, coefs_new = self.vectro_to_tuples(points, coefs, sigma * direction)
        j_tilde_new = self.j_tilde(points_new, coefs_new)
        while j_tilde_new - j_tilde_init > sigma * desired_descent:
            sigma *= beta
            points_new, coefs_new = self.vectro_to_tuples(
                points, coefs, sigma * direction
            )
            j_tilde_new = self.j_tilde(points_new, coefs_new)
        return points_new, coefs_new

    def globalized_newton_step(
        self,
        points: np.ndarray,
        coefs: np.ndarray,
    ) -> tuple:
        newton_acceptance = 1e-1
        descent_constant_1 = 1e-2
        descent_constant_2 = 1e-2

        grad_j_z = self.grad_j(points, coefs)
        hess_j_z = self.hess_j(points, coefs)
        try:
            update_direction = np.linalg.solve(hess_j_z, -grad_j_z)
            update_norm = np.linalg.norm(update_direction)
            grad_direction = -grad_j_z @ update_direction
            condition = (
                grad_direction
                >= min(
                    descent_constant_1,
                    descent_constant_2 * update_norm**newton_acceptance,
                )
                * update_norm**2
            )
            if not condition:
                raise np.linalg.LinAlgError("Insufficient descent in Newton direction")
            choice = "Newton"
        except np.linalg.LinAlgError:
            update_direction = -grad_j_z
            choice = "Gradient"
        points_new, coefs_new = self.armijo(points, coefs, update_direction, grad_j_z)
        return points_new, coefs_new, choice

    def lgcg_step(
        self, p_u: Callable, u: Measure, epsilon: float, q_u: float, radii: np.ndarray
    ) -> tuple:
        j_initial = self.j(u)
        condition = False
        if len(radii):
            radius = np.max(radii)
        else:
            radius = self.max_radius
        x_k, found_points, found_values, global_valid = self.global_search(
            u, epsilon, q_u, p_u, radius
        )
        Phi = self.M * max((np.abs(p_u(x_k))[0] - self.alpha), 0) + q_u
        if Phi > q_u:
            v = Measure(found_points, self.M * np.sign(found_values))
            # v = Measure([x_k], [self.M * np.sign(p_u(x_k)[0])])
        else:
            v = Measure()
        updates = 0
        self.C_raw /= 2
        while not condition:
            # Increase C_raw until it satisfies the descent condition
            updates += 1
            self.C_raw *= 2
            Curv = self.C_raw * self.M**2
            eta = min(1, Phi / Curv)
            u_plus = u * (1 - eta) + v * eta
            jdiff = self.j(u_plus) - j_initial
            if Phi <= Curv:
                expected_decrease = -0.5 * Phi**2 / Curv
            else:
                expected_decrease = 0.5 * Curv - Phi
            if abs(expected_decrease) < self.machine_precision:
                condition = True
            else:
                condition = jdiff <= expected_decrease
        if updates < 2:
            # There has been no increase of the curvature constant, try a smaller value
            while condition and self.C_raw >= self.C_0:
                previous_u_plus = u_plus * 1
                self.C_raw /= 2
                Curv = self.C_raw * self.M**2
                eta = min(1, Phi / Curv)
                u_plus = u * (1 - eta) + v * eta
                jdiff = self.j(u_plus) - j_initial
                if Phi <= Curv:
                    expected_decrease = -0.5 * Phi**2 / Curv
                else:
                    expected_decrease = 0.5 * Curv - Phi
                if abs(expected_decrease) < self.machine_precision:
                    condition = True
                else:
                    condition = jdiff <= expected_decrease
            self.C_raw *= 2
            u_plus = previous_u_plus * 1
        if not global_valid:
            # We have a global maximum x_k
            epsilon = 0.5 * Phi / self.M
        return u_plus, epsilon, global_valid

    def domain_tests(
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
        output_bools.append(np.linalg.norm(coefs_new, ord=1) <= 2 * self.M)

        # Sign test
        if np.all(np.sign(coefs_new) == np.sign(coefs)):
            output_bools.append(True)
        else:
            output_bools.append(False)

        # # Descent test
        # j_tilde_diff = self.j_tilde(points_new, coefs_new) - self.j_tilde(points, coefs)
        # if j_tilde_diff >= -self.machine_precision:
        #     output_bools.append(False)
        # else:
        #     output_bools.append(True)
        #     logging.info(j_tilde_diff)

        return output_bools

    def descent_test(
        self, points: np.ndarray, coefs: np.ndarray, epsilon: float, radii: list
    ) -> bool:
        grad_j_z = self.grad_j(points, coefs)
        points_part = 0
        for i, radius in enumerate(radii):
            points_part += radius * np.linalg.norm(
                grad_j_z[i * len(points[0]) : (i + 1) * len(points[0])]
            )
        grad_coefs = grad_j_z[len(points.flatten()) :]
        gap = (
            points_part
            + self.M * abs(min(0, np.min(np.multiply(grad_coefs, np.sign(coefs)))))
            + grad_coefs @ coefs
        )
        if epsilon <= self.C_raw * self.M:
            ineq = gap >= 0.5 * epsilon**2 / self.C_raw
        else:
            ineq = gap >= (2 * self.M * epsilon - self.C_raw * self.M**2) / 2

        if not ineq:
            logging.info(f"gap: {gap}, norm grad: {np.linalg.norm(grad_j_z)}")
        return ineq

    def compute_radii(self, u: Measure) -> list:
        radii = []
        if not len(u.coefficients):
            return radii
        grad_P = self.grad_P(u)
        hess_P = self.hess_P(u)
        grads = grad_P(u.support)
        hesses = hess_P(u.support)
        for point_grad, point_hess in zip(grads, hesses):
            eigenvalue = np.min(np.abs(np.linalg.eigvals(point_hess)))
            local_radius = 4 * np.linalg.norm(point_grad) / eigenvalue
            radii.append(min(local_radius, self.max_radius))
        logging.info(f"Radii: {radii}")
        return radii

    def nlgcg(
        self,
        tol: float,
        max_radius: float,
        drop_frequency: int = 5,
        u_0: Measure = Measure(),
        Psi_0: float = 1,
    ) -> tuple:
        self.max_radius = max_radius
        self.M = self.M_0
        self.C_raw = self.C_0
        epsilon = 0.5 * self.j(u_0) / self.M
        Psi_k = min(Psi_0, self.Psi_0)
        k = 0
        dropped = False
        optimal = False
        dropped_tot = 0
        initial_time = time.time()
        times = [time.time() - initial_time]
        supports = [0]
        inner_loop = [0]
        objective_values = [self.j(u_0)]
        epsilons = [epsilon]
        lgcg_lazy = 0
        lgcg_total = 0

        u_plus = u_0 * 1
        while 2 * self.M * epsilon > tol:
            global_valid = "N/A"
            u = u_plus * 1
            points, coefs, u_ks, radii = self.local_merging_update_radii(u)
            local_M = self.j(u_ks) / self.alpha
            epsilon_ks = epsilon + 0.5 * (self.j(u_ks) - self.j(u)) / self.M
            p_u_ks = self.p(u_ks)
            q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
            u_ks_gcg = u_ks * 1
            u_ks_new = u_ks * 1
            u_lm = u_ks * 1

            s = 1
            while len(u_ks.coefficients):
                # Inner loop
                points_new, coefs_new, newton_choice = self.globalized_newton_step(
                    points, coefs
                )
                u_ks_new = Measure(points_new, coefs_new)

                # Monitor statistics
                times.append(time.time() - initial_time)
                supports.append(len(u_ks.support))
                inner_loop.append(1)
                objective_values.append(self.j(u_ks))
                epsilons.append(epsilon_ks)
                logging.info(
                    f"{k}, {s}: Globalization: {newton_choice}, support: {len(u_ks.support)}, epsilon: {epsilon_ks}, c_raw: {self.C_raw} objective: {self.j(u_ks):.6E}"
                )

                # Check validity of the newton step
                descent_test = self.descent_test(points, coefs, epsilon_ks, radii)
                if not descent_test:
                    u_ks_gcg, epsilon_ks, global_valid = self.lgcg_step(
                        p_u_ks, u_ks, epsilon_ks, q_u_ks, radii
                    )
                    lgcg_lazy += int(global_valid)
                    lgcg_total += 1
                    descent_test = self.descent_test(points, coefs, epsilon_ks, radii)
                else:
                    global_valid = "N/A"
                if 2 * local_M * epsilon_ks <= tol:  # Optimality reached
                    optimal = True
                    s += 1
                    break
                else:
                    optimal = False
                if not descent_test:
                    s += 1
                    logging.info(f"Descent test: {descent_test}")
                    break
                domain_tests = self.domain_tests(points, coefs, points_new, coefs_new)
                if not all(domain_tests):
                    s += 1
                    logging.info(f"Domain tests: {domain_tests}")
                    break

                # Perform drop and local merging
                if (s + 1) % drop_frequency == 0:
                    u_ks_drop, dropped = self.drop_step(u_ks_new)
                    dropped_tot += dropped
                    points, coefs, u_ks, radii = self.local_merging_update_radii(
                        u_ks_drop
                    )
                    epsilon_ks = (
                        epsilon_ks + 0.5 * (self.j(u_ks) - self.j(u_ks_drop)) / self.M
                    )
                else:
                    u_ks = u_ks_new * 1
                    points = points_new.copy()
                    coefs = coefs_new.copy()
                local_M = self.j(u_ks) / self.alpha
                p_u_ks = self.p(u_ks)
                q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
                s += 1

            if optimal:
                u = u_ks * 1
                break

            all_iterates = [u, u_lm, u_ks, u_ks_new, u_ks_gcg]
            iterate_values = [self.j(iterate) for iterate in all_iterates]
            choice_index = np.argmin(iterate_values)
            u = all_iterates[choice_index] * 1

            p_u = self.p(u)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)

            u_gcg, epsilon, global_valid = self.lgcg_step(p_u, u, epsilon, q_u, radii)
            lgcg_lazy += int(global_valid)
            lgcg_total += 1

            if len(u_gcg.coefficients):
                u_drop, dropped = self.drop_step(u_gcg)
                u_plus, finite_psi = self.finite_dimensional_step(
                    u_drop, self.machine_precision, mode="positive"
                )
                dropped_tot += dropped
            else:
                u_plus = u_gcg * 1
            self.M = self.j(u) / self.alpha
            # Psi_k = max(Psi_k / 2, self.machine_precision)

            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            inner_loop.append(0)
            objective_values.append(self.j(u))
            epsilons.append(epsilon)
            logging.info(
                f"{k}: choice: {choice_index}, lazy: {global_valid}, support: {len(u.support)}, epsilon: {epsilon}, c_raw: {self.C_raw}, objective: {self.j(u):.6E}"
            )
            logging.info(
                "============================================================================================="
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
