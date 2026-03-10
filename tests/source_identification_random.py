import numpy as np
import sys
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))
from lib.measure import Measure
from lib.fista import FISTA
from lib.ssn import SSN
from lib.cvxpy_solver import CVXPY
from lib.sklearn_solver import SKLEARN
from lazified_pdap_finite import LazifiedPDAPFinite
from lazified_pdap import LazifiedPDAP

results_dir = Path("results/source_identification_random")
results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
Omega_size = Omega[0][1] - Omega[0][0]
alpha = 1e-2
observation_resolution = 4
std_factor = 0.1
gamma = 1
theta = 1e-1
sigma = 2e-3
m = 1e-3
bar_m = 1e-1
L = 1
R = 5e-2


def generate_function(true_sources: np.ndarray, true_weights: np.ndarray) -> tuple:
    observations = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(bound[0], bound[1], observation_resolution + 2)
                    for bound in Omega
                )
            )
        )
        .reshape(len(Omega), -1)
        .T
    )
    observations = np.array(
        [obs for obs in observations if all(obs != 0) and all(obs != 1)]
    )

    def kernel(x):
        # Input is 2D array of shape (number of points, Omega dimension)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        columns = []
        outer_factor = np.sqrt(std_factor * np.pi) ** Omega.shape[0]
        for point in observations:
            diff = point - x  # (len(x), Omega.shape[0])
            norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
            exponentiated = np.exp(norms)  # (len(x),)
            columns.append(exponentiated)
        result = (
            np.transpose(np.array(columns), axes=(1, 0)) / outer_factor
        )  # shape=(len(x), len(observations))
        return result

    def grad_kernel(x):
        # Input is 2D array of shape (number of points, Omega dimension)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        gradients = []
        outer_factor = std_factor * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 2
        for point in observations:
            diff = point - x  # (len(x), Omega.shape[0])
            norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
            exponentiated = np.exp(norms)  # (len(x),)
            gradient = diff * exponentiated.reshape(
                -1, 1
            )  # shape=(len(x),Omega.shape[0])
            gradients.append(gradient)
        result = (
            np.transpose(np.array(gradients), axes=(1, 0, 2)) / outer_factor
        )  # The Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0])
        return result

    def hess_kernel(x):
        # Input is 2D array of shape (number of points, Omega dimension)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        hessians = []
        outer_factors = [
            -std_factor * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 2,
            std_factor**2 * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 4,
        ]
        for point in observations:
            diff = point - x  # (len(x), Omega.shape[0])
            norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
            exponentiated_normed_1 = np.exp(norms) / outer_factors[0]  # (len(x),)
            exponentiated_normed_2 = np.exp(norms) / outer_factors[1]  # (len(x),)
            first_part = np.repeat(
                np.eye(Omega.shape[0])[np.newaxis, :], len(x), axis=0
            ) * exponentiated_normed_1.reshape(
                -1, 1, 1
            )  # shape=(len(x),Omega.shape[0],Omega.shape[0])
            second_part = np.einsum(
                "ij,ik->ijk", diff, diff
            ) * exponentiated_normed_2.reshape(
                -1, 1, 1
            )  # shape=(len(x),Omega.shape[0],Omega.shape[0])
            hessians.append(first_part + second_part)
        result = np.transpose(
            np.array(hessians), axes=(1, 0, 2, 3)
        )  # The derivative of the Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0], Omega.shape[0])
        return result

    u_hat = Measure(support=true_sources, coefficients=true_weights)
    target = u_hat.duality_pairing(kernel)

    g = lambda u: alpha * np.linalg.norm(u, ord=1)
    f = lambda u: 0.5 * np.linalg.norm(u.duality_pairing(kernel) - target) ** 2
    j = lambda u: f(u) + g(u)

    def p(u):
        Ku = u.duality_pairing(kernel)
        inner = Ku - target
        return lambda x: -kernel(x) @ inner

    def grad_P(u):
        p_u = p(u)
        inner = target - u.duality_pairing(kernel)
        return lambda x: np.sign(p_u(x)).reshape(-1, 1) * np.tensordot(
            grad_kernel(x), inner, axes=([1, 0])
        )

    def hess_P(u):
        p_u = p(u)
        inner = target - u.duality_pairing(kernel)
        return lambda x: np.sign(p_u(x)).reshape(-1, 1, 1) * np.tensordot(
            hess_kernel(x), inner, axes=([1, 0])
        )

    def grad_j(positions, coefs):
        K_matrix = kernel(positions)
        grad_F = (K_matrix.T @ coefs).flatten() - target
        nabla_x = coefs.reshape(-1, 1) * np.tensordot(
            grad_kernel(positions), grad_F, axes=([1, 0])
        )
        nabla_u = np.dot(K_matrix, grad_F) + alpha * np.sign(coefs)
        return np.append(nabla_x.flatten(), nabla_u, axis=0).flatten()

    def hess_j(positions, coefs):
        kappa_values = kernel(positions)
        grad_kappa_values = grad_kernel(positions)
        hess_kappa_values = hess_kernel(positions)
        matrix_dimension = len(positions) * Omega.shape[0] + len(coefs)
        hesse_matrix = np.zeros((matrix_dimension, matrix_dimension))
        step = Omega.shape[0]
        coefs_delay = step * len(positions)
        inner = (kappa_values.T @ coefs).flatten() - target
        for i in range(len(positions)):
            # nabla_{x_i,x_j}
            for j in range(len(positions)):
                if j < i:
                    continue
                block = (
                    coefs[i]
                    * coefs[j]
                    * np.matmul(grad_kappa_values[i].T, grad_kappa_values[j])
                )
                if i == j:
                    block += coefs[i] * np.tensordot(
                        hess_kappa_values[i], inner, axes=([0, 0])
                    )
                hesse_matrix[i * step : (i + 1) * step, j * step : (j + 1) * step] = (
                    block
                )
                hesse_matrix[j * step : (j + 1) * step, i * step : (i + 1) * step] = (
                    block.T
                )
            # nabla_{x_i,u_j}
            for j in range(len(coefs)):
                block = coefs[i] * np.matmul(grad_kappa_values[i].T, kappa_values[j])
                if i == j:
                    block += np.matmul(grad_kappa_values[i].T, inner)
                hesse_matrix[i * step : (i + 1) * step, coefs_delay + j] = block
                hesse_matrix[coefs_delay + j, i * step : (i + 1) * step] = block.T
        for i in range(len(coefs)):
            # nabla_{u_i,u_j}
            for j in range(len(coefs)):
                if j < i:
                    continue
                block = np.dot(kappa_values[i], kappa_values[j])
                hesse_matrix[coefs_delay + i, coefs_delay + j] = block
                hesse_matrix[coefs_delay + j, coefs_delay + i] = block
        return hesse_matrix

    a = np.arange(0, 1, 0.01)
    if Omega.shape[0] == 1:
        vals = np.linalg.norm(kernel(a.reshape(1, -1).T), axis=1)
        norm_kernel = max(vals)
    elif Omega.shape[0] == 2:
        x, y = np.meshgrid(a, a)
        points = np.array(list(zip(x.flatten(), y.flatten())))
        vals = np.linalg.norm(kernel(points), axis=1)
        norm_kernel = max(vals)  # ||k([0.5,0.5])||
    elif Omega.shape[0] == 3:
        x, y, z = np.meshgrid(a, a, a)
        points = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))
        vals = np.linalg.norm(kernel(points), axis=1)
        norm_kernel = max(vals)
    else:
        norm_kernel = 10

    a = np.arange(0, 1, 0.1)
    if Omega.shape[0] == 1:
        vals = np.linalg.norm(grad_kernel(a.reshape(1, -1).T), axis=(1, 2))
        norm_kernel1 = max(norm_kernel, max(vals))
    elif Omega.shape[0] == 2:
        B, D = np.meshgrid(a, a)
        points = np.array(list(zip(B.flatten(), D.flatten())))
        vals = np.linalg.norm(grad_kernel(points), axis=(1, 2))
        norm_kernel1 = max(norm_kernel, max(vals))
    elif Omega.shape[0] == 3:
        x, y, z = np.meshgrid(a, a, a)
        points = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))
        vals = np.linalg.norm(grad_kernel(points), axis=(1, 2))
        norm_kernel1 = max(norm_kernel, max(vals))
    else:
        norm_kernel1 = 50

    return (
        target,
        kernel,
        g,
        f,
        p,
        grad_P,
        hess_P,
        norm_kernel,
        norm_kernel1,
        grad_j,
        hess_j,
    )


def compute_intervals(inner_loop):
    intervals = []
    current_inner = False
    for ind, i in enumerate(inner_loop):
        if i:
            if not current_inner:
                start = ind - 0.5
                current_inner = True
        else:
            if current_inner:
                end = ind - 0.5
                intervals.append((start, end))
                current_inner = False
    if inner_loop[-1]:
        end = len(inner_loop) - 0.5
        intervals.append((start, end))
    return np.array(intervals)


def get_grid(size: int) -> np.ndarray:
    grid = (
        np.array(
            np.meshgrid(
                *(np.linspace(bound[0], bound[1], size + 1)[1:] for bound in Omega)
            )
        )
        .reshape(len(Omega), -1)
        .T
    )
    return grid


def sample_domain(
    size: int, domain: np.ndarray, rng: np.random._generator.Generator
) -> np.ndarray:
    # Generate a uniform sample of shape (size,domain.shape[0]) in the given domain
    columns = []
    for i, bounds in enumerate(domain):
        if i == 0:
            columns.append(rng.random((size, 1)) * (bounds[1] - bounds[0]) + bounds[0])
        else:
            columns.append(rng.random((size, 1)) * (bounds[1] - bounds[0]) + bounds[0])
    sample = np.concatenate(columns, axis=1)
    return sample


def adapt_time(times, residuals, frame=100, resolution=1):
    to_return = []
    last_pos = 0
    last_res = residuals[0]
    for t in range(int(frame / resolution)):
        minimim_time = t * resolution
        maximum_time = (t + 1) * resolution
        added = False
        for i, (res, tim) in enumerate(zip(residuals[last_pos:], times[last_pos:])):
            if tim < maximum_time and tim >= minimim_time:
                to_return.append(res)
                last_res = res
                last_pos += i + 1
                added = True
                break
        if not added:
            to_return.append(last_res)
        if t * resolution >= times[-1]:
            break
    to_return.append(residuals[-1])
    return to_return


def bring_to_same_length(arrays):
    max_length = max(len(arr) for arr in arrays)
    new_arrays = []
    for arr in arrays:
        if len(arr) < max_length:
            last_val = arr[-1]
            arr = list(arr) + [last_val] * (max_length - len(arr))
        new_arrays.append(np.array(arr))
    return new_arrays


def plot_observations(true_measure: Measure, predicted_measure: Measure, iter: int):
    # 2D problem with both true and predicted sources
    logging.getLogger().setLevel(logging.WARNING)  # Supress logging
    resolution = 100
    a_1 = np.linspace(Omega[0][0], Omega[0][1], resolution, endpoint=False)
    a_2 = np.linspace(Omega[1][0], Omega[1][1], resolution, endpoint=False)
    x, y = np.meshgrid(a_1, a_2)
    points = np.array(list(zip(x.flatten(), y.flatten())))

    def plot_kernel(x):
        # Input is 2D array of shape (number of points, Omega dimension)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        columns = []
        outer_factor = np.sqrt(std_factor * np.pi) ** Omega.shape[0]
        for point in points:
            diff = point - x  # (len(x), Omega.shape[0])
            norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
            exponentiated = np.exp(norms)  # (len(x),)
            columns.append(exponentiated)
        result = (
            np.transpose(np.array(columns), axes=(1, 0)) / outer_factor
        )  # shape=(len(x), len(observations))
        return result

    true_vals = true_measure.duality_pairing(plot_kernel).reshape((100, 100))
    pred_vals = predicted_measure.duality_pairing(plot_kernel).reshape((100, 100))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    contour1 = ax1.contourf(x, y, true_vals, levels=100)
    fig.colorbar(contour1, ax=ax1)
    contour2 = ax2.contourf(x, y, pred_vals, levels=100)
    fig.colorbar(contour2, ax=ax2)
    contour3 = ax3.contourf(x, y, np.abs(pred_vals - true_vals), levels=100)
    fig.colorbar(contour3, ax=ax3)
    for i, x in enumerate(true_measure.support):
        if true_measure.coefficients[i] < 0:
            color = "b"
        else:
            color = "r"
        ax1.plot([x[0]], [x[1]], "P", c=color, alpha=0.5)
        ax1.add_patch(
            plt.Circle(
                (x[0], x[1]),
                radius=true_measure.coefficients[i] / 10,
                color=color,
                fill=False,
                alpha=0.5,
            )
        )
        ax3.plot([x[0]], [x[1]], "P", c=color, alpha=0.5)

    for i, x in enumerate(predicted_measure.support):
        if predicted_measure.coefficients[i] < 0:
            color = "b"
        else:
            color = "r"
        ax2.plot([x[0]], [x[1]], "o", c=color, alpha=0.5)
        ax2.add_patch(
            plt.Circle(
                (x[0], x[1]),
                radius=predicted_measure.coefficients[i] / 10,
                color=color,
                fill=False,
                alpha=0.5,
            )
        )
        ax3.plot([x[0]], [x[1]], "o", c=color, alpha=0.5)

    ax1.set_xlabel("True heat distribution")
    ax2.set_xlabel("Predicted heat distribution")
    ax3.set_xlabel("Absolute error in heat distribution")

    ax1.set_xlim(Omega[0][0], Omega[0][1])
    ax1.set_ylim(Omega[1][0], Omega[1][1])
    ax2.set_xlim(Omega[0][0], Omega[0][1])
    ax2.set_ylim(Omega[1][0], Omega[1][1])
    ax3.set_xlim(Omega[0][0], Omega[0][1])
    ax3.set_ylim(Omega[1][0], Omega[1][1])

    plt.savefig(results_dir / f"heat_distribution_{iter}.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


def experiment():
    rng = np.random.default_rng(seed=42)
    loops = 5
    resolution = 0.1
    source_nbr = 10

    all_times_pdap = []
    all_times_nlgcg = []
    all_times_lpdap = []
    all_res_pdap = []
    all_res_nlgcg = []
    all_res_lpdap = []
    all_supp_pdap = []
    all_supp_nlgcg = []
    all_supp_lpdap = []
    all_sklearn_solutions_default = []
    all_sklearn_solutions_custom = []
    all_fista_solutions = []
    all_flpdap_solutions = []

    for _ in range(loops):
        logging.info("=" * 50)
        logging.info(f"Loop {_+1}")

        true_sources = []
        while len(true_sources) < source_nbr:
            point = sample_domain(1, Omega, rng).flatten()
            if not len(true_sources):
                true_sources.append(point)
            elif not any(
                np.linalg.norm(point - source) < 4 * R for source in true_sources
            ) and not any(
                np.min([np.abs(poi - bound[0]), np.abs(poi - bound[1])]) < 2 * R
                for poi, bound in zip(point, Omega)
            ):
                true_sources.append(point)
        true_sources = np.array(true_sources)
        true_weights_raw = sample_domain(source_nbr, np.array([[-1, 1]]), rng).flatten()
        weight_signs = np.sign(true_weights_raw)
        true_weights = weight_signs * (np.sqrt(np.abs(true_weights_raw)))
        L_H = max(np.abs(true_weights))
        (
            target,
            kernel,
            g,
            f,
            p,
            grad_P,
            hess_P,
            norm_kernel,
            norm_kernel1,
            grad_j,
            hess_j,
        ) = generate_function(true_sources, true_weights)

        exp = LazifiedPDAP(
            target=target,
            kernel=kernel,
            g=g,
            f=f,
            p=p,
            grad_P=grad_P,
            hess_P=hess_P,
            norm_kernel=norm_kernel,
            norm_kernel1=norm_kernel1,
            grad_j=grad_j,
            hess_j=hess_j,
            alpha=alpha,
            Omega=Omega,
            gamma=gamma,
            theta=theta,
            sigma=sigma,
            m=m,
            bar_m=bar_m,
            L=L,
            L_H=L_H,
            R=R,
        )

        # PDAP
        logging.info(f"Computing PDAP solution")
        u_pdap, P_values_pdap, times_pdap, supports_pdap, objective_values_pdap = (
            exp.pdap(tol=1e-12, do_logging=False)
        )
        logging.info(
            "-------------------------------------------------------------------"
        )

        # NLPDAP
        logging.info(f"Computing NLPDAP solution")
        (
            u_nlgcg,
            times_nlgcg,
            supports_nlgcg,
            inner_loop_nlgcg,
            lgcg_lazy_nlgcg,
            lgcg_total_nlgcg,
            objective_values_nlgcg,
            dropped_tot_nlgcg,
            epsilons_nlgcg,
        ) = exp.newton(tol=1e-12, damped=False, do_logging=False)
        intervals_nlgcg = compute_intervals(inner_loop_nlgcg)
        logging.info(f"Lazy LGCG steps: {lgcg_lazy_nlgcg}")
        logging.info(f"Total LGCG steps: {lgcg_total_nlgcg}")
        logging.info(
            "-------------------------------------------------------------------"
        )

        # LPDAP
        logging.info(f"Computing LPDAP solution")
        (
            u_lpdap,
            intermediate_u_lpdap,
            Phi_ks_lpdap,
            times_lpdap,
            supports_lpdap,
            objective_values_lpdap,
            dropped_tot_lpdap,
            epsilons_lpdap,
        ) = exp.lpdap(tol=1e-12, do_logging=False)
        logging.info(
            "-------------------------------------------------------------------"
        )

        plot_observations(
            Measure(support=true_sources, coefficients=true_weights), u_nlgcg, _ + 1
        )

        optimum = (
            min(
                [
                    objective_values_pdap[-1],
                    objective_values_lpdap[-1],
                    objective_values_nlgcg[-1],
                ]
            )
            - 1e-14
        )

        sklearn_solutions_default = {}
        sklearn_solutions_custom = {}
        fista_solutions = {}
        flpdap_solutions = {}
        sizes = [10, 32, 100]
        for size in sizes:
            grid = get_grid(size)
            K_transpose = kernel(grid)

            logging.info(
                f"Solving with scikit-learn on uniform grid of size {int(size**2)} with default parameters"
            )
            sklearn_exp = SKLEARN(K=K_transpose.T, alpha=alpha, target=target)
            u_sklearn, objective_value_sklearn, time_sklearn = sklearn_exp.solve()
            sklearn_solutions_default[size] = (
                u_sklearn,
                objective_value_sklearn - optimum,
                time_sklearn,
            )
            logging.info(
                "-------------------------------------------------------------------"
            )

            logging.info(
                f"Solving with scikit-learn on uniform grid of size {int(size**2)} with tol=1e-6, max_iter=10000"
            )
            sklearn_exp = SKLEARN(K=K_transpose.T, alpha=alpha, target=target)
            u_sklearn, objective_value_sklearn, time_sklearn = sklearn_exp.solve(
                tol=1e-6, max_iter=10000
            )
            sklearn_solutions_custom[size] = (
                u_sklearn,
                objective_value_sklearn - optimum,
                time_sklearn,
            )
            logging.info(
                "-------------------------------------------------------------------"
            )

            logging.info(f"Solving with FISTA on uniform grid of size {int(size**2)}")
            fista_exp = FISTA(K=K_transpose.T, alpha=alpha, target=target)
            u_fista, objective_values_fista, times_fista = fista_exp.solve(
                max_time=300, do_logging=False
            )
            fista_solutions[size] = (
                u_fista,
                adapt_time(
                    times_fista,
                    objective_values_fista - optimum,
                    frame=300,
                    resolution=resolution,
                ),
                times_fista,
                objective_values_fista - optimum,
            )
            logging.info(
                "-------------------------------------------------------------------"
            )

            logging.info(
                f"Solving with finite LPDAP on uniform grid of size {int(size**2)}"
            )
            flpdap_exp = LazifiedPDAPFinite(
                K_transpose=K_transpose, alpha=alpha, target=target
            )
            u_flpdap, objective_values_flpdap, times_flpdap = flpdap_exp.solve(
                tol=1e-10, do_logging=False
            )
            flpdap_solutions[size] = (
                u_flpdap,
                adapt_time(
                    times_flpdap,
                    objective_values_flpdap - optimum,
                    frame=300,
                    resolution=resolution,
                ),
                times_flpdap,
                objective_values_flpdap - optimum,
            )
            logging.info(
                "-------------------------------------------------------------------"
            )

        all_sklearn_solutions_default.append(sklearn_solutions_default)
        all_sklearn_solutions_custom.append(sklearn_solutions_custom)
        all_fista_solutions.append(fista_solutions)
        all_flpdap_solutions.append(flpdap_solutions)
        all_times_pdap.append(times_pdap)
        all_times_nlgcg.append(times_nlgcg)
        all_times_lpdap.append(times_lpdap)
        all_res_pdap.append(
            adapt_time(
                times_pdap,
                objective_values_pdap - optimum,
                frame=300,
                resolution=resolution,
            )
        )
        all_res_nlgcg.append(
            adapt_time(
                times_nlgcg,
                objective_values_nlgcg - optimum,
                frame=300,
                resolution=resolution,
            )
        )
        all_res_lpdap.append(
            adapt_time(
                times_lpdap,
                objective_values_lpdap - optimum,
                frame=300,
                resolution=resolution,
            )
        )
        all_supp_pdap.append(supports_pdap)
        all_supp_nlgcg.append(supports_nlgcg)
        all_supp_lpdap.append(supports_lpdap)

    for size in sizes:
        residual_flpdap = np.mean([d[size][1][-1] for d in all_flpdap_solutions])
        time_flpdap = np.mean([d[size][2][-1] for d in all_flpdap_solutions])
        logging.info(
            f"Finite LPDAP on grid of size {size**2}: mean absolute residual {residual_flpdap:.12E}, mean time {time_flpdap:.3f}s"
        )
        residual_sklearn = np.mean([d[size][1] for d in all_sklearn_solutions_default])
        time_sklearn = np.mean([d[size][2] for d in all_sklearn_solutions_default])
        logging.info(
            f"Scikit-learn default on grid of size {size**2}: mean absolute residual {residual_sklearn:.12E}, mean grid residual {residual_sklearn - residual_flpdap:.12E}, mean time {time_sklearn:.3f}s"
        )
        residual_sklearn = np.mean([d[size][1] for d in all_sklearn_solutions_custom])
        time_sklearn = np.mean([d[size][2] for d in all_sklearn_solutions_custom])
        logging.info(
            f"Scikit-learn custom on grid of size {size**2}: mean absolute residual {residual_sklearn:.12E}, mean grid residual {residual_sklearn - residual_flpdap:.12E}, mean time {time_sklearn:.3f}s"
        )

    pdap_residuals_mean = np.mean(bring_to_same_length(all_res_pdap), axis=0)
    pdap_residuals_ste = np.std(bring_to_same_length(all_res_pdap), axis=0) / np.sqrt(
        loops
    )
    nlgcg_residuals_mean = np.mean(bring_to_same_length(all_res_nlgcg), axis=0)
    nlgcg_residuals_ste = np.std(bring_to_same_length(all_res_nlgcg), axis=0) / np.sqrt(
        loops
    )
    lpdap_residuals_mean = np.mean(bring_to_same_length(all_res_lpdap), axis=0)
    lpdap_residuals_ste = np.std(bring_to_same_length(all_res_lpdap), axis=0) / np.sqrt(
        loops
    )
    pdap_supports_mean = np.mean(bring_to_same_length(all_supp_pdap), axis=0)
    pdap_supports_ste = np.std(bring_to_same_length(all_supp_pdap), axis=0) / np.sqrt(
        loops
    )
    nlgcg_supports_mean = np.mean(bring_to_same_length(all_supp_nlgcg), axis=0)
    nlgcg_supports_ste = np.std(bring_to_same_length(all_supp_nlgcg), axis=0) / np.sqrt(
        loops
    )
    lpdap_supports_mean = np.mean(bring_to_same_length(all_supp_lpdap), axis=0)
    lpdap_supports_ste = np.std(bring_to_same_length(all_supp_lpdap), axis=0) / np.sqrt(
        loops
    )

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["PDAP", "LPDAP", "NLGCG"]
    styles = ["-", "--", "-."]
    colors = ["blue", "orange", "green"]
    for array, array_ste, name, style, color in zip(
        [pdap_supports_mean, lpdap_supports_mean, nlgcg_supports_mean],
        [pdap_supports_ste, lpdap_supports_ste, nlgcg_supports_ste],
        names,
        styles,
        colors,
    ):
        ax.plot(np.arange(len(array)), array, style, label=name)
        ax.fill(
            np.hstack(
                (
                    np.arange(len(array)),
                    np.arange(len(array))[::-1],
                )
            ),
            np.hstack(
                (
                    np.array(array) - np.array(array_ste),
                    np.array(array)[::-1] + np.array(array_ste)[::-1],
                )
            ),
            color,
            alpha=0.3,
        )
    plt.ylabel("Support size")
    plt.xlabel("Total iterations")
    ax.legend()
    plt.savefig(results_dir / "support_iter.png", bbox_inches="tight")
    plt.close()

    # Plot residuals in time
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["PDAP", "LPDAP", "NLGCG"]
    styles = ["-", "-.", "--"]
    colors = ["blue", "orange", "green"]
    for array, array_ste, name, style, color in zip(
        [pdap_residuals_mean, lpdap_residuals_mean, nlgcg_residuals_mean],
        [pdap_residuals_ste, lpdap_residuals_ste, nlgcg_residuals_ste],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            np.arange(len(array)) * resolution, array, style, label=name, color=color
        )
        ax.fill(
            np.hstack(
                (
                    np.arange(len(array)) * resolution,
                    np.arange(len(array))[::-1] * resolution,
                )
            ),
            np.hstack(
                (
                    np.maximum(
                        np.array(array) - np.array(array_ste),
                        1e-13,
                    ),
                    np.array(array)[::-1] + np.array(array_ste)[::-1],
                )
            ),
            color,
            alpha=0.3,
        )
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-12, 30)
    ax.legend()
    plt.savefig(results_dir / "res_time.png", bbox_inches="tight")
    plt.close()

    # Plot FLPDAP vs FISTA residuals on grid in time
    names = (
        [f"FISTA, {size}x{size} grid" for size in sizes]
        + [f"Discretized LPDPA, {size}x{size} grid" for size in sizes]
        + [f"LPDAP, gridless"]
    )
    colors = ["blue"] * len(sizes) + ["green"] * len(sizes) + ["black"]
    styles = ["-", "--", "-."] + ["-", "--", "-."] + ["-"]
    residuals_flpdap_mean = [
        np.mean(
            bring_to_same_length([d[size][1] for d in all_flpdap_solutions]), axis=0
        )
        for size in sizes
    ]
    residuals_flpdap_ste = [
        np.std(bring_to_same_length([d[size][1] for d in all_flpdap_solutions]), axis=0)
        / np.sqrt(loops)
        for size in sizes
    ]
    residuals_fista_mean = [
        np.mean(bring_to_same_length([d[size][1] for d in all_fista_solutions]), axis=0)
        for size in sizes
    ]
    residuals_fista_ste = [
        np.std(bring_to_same_length([d[size][1] for d in all_fista_solutions]), axis=0)
        / np.sqrt(loops)
        for size in sizes
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for array, array_ste, name, style, color in zip(
        residuals_fista_mean + residuals_flpdap_mean + [lpdap_residuals_mean],
        residuals_fista_ste + residuals_flpdap_ste + [lpdap_residuals_ste],
        names,
        styles,
        colors,
    ):
        ax.loglog(
            np.arange(len(array)) * resolution, array, style, label=name, color=color
        )
        ax.fill(
            np.hstack(
                (
                    np.arange(len(array)) * resolution,
                    np.arange(len(array))[::-1] * resolution,
                )
            ),
            np.hstack(
                (
                    np.maximum(
                        np.array(array) - np.array(array_ste),
                        1e-13,
                    ),
                    np.array(array)[::-1] + np.array(array_ste)[::-1],
                )
            ),
            color,
            alpha=0.3,
        )
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-10, 100)
    ax.legend()
    plt.savefig(results_dir / "grid_res_time.png", bbox_inches="tight")
    plt.close()

    # Plot FLPDAP vs FISTA residuals on grid in iters
    names = (
        [f"FISTA, {size}x{size} grid" for size in sizes]
        + [f"Discretized LPDPA, {size}x{size} grid" for size in sizes]
        + [f"LPDAP, gridless"]
    )
    colors = ["blue"] * len(sizes) + ["green"] * len(sizes) + ["black"]
    styles = ["-", "--", "-."] + ["-", "--", "-."] + ["-"]
    residuals_flpdap_mean = [
        np.mean(
            bring_to_same_length([d[size][3] for d in all_flpdap_solutions]), axis=0
        )
        for size in sizes
    ]
    residuals_flpdap_ste = [
        np.std(bring_to_same_length([d[size][3] for d in all_flpdap_solutions]), axis=0)
        / np.sqrt(loops)
        for size in sizes
    ]
    residuals_fista_mean = [
        np.mean(bring_to_same_length([d[size][3] for d in all_fista_solutions]), axis=0)
        for size in sizes
    ]
    residuals_fista_ste = [
        np.std(bring_to_same_length([d[size][3] for d in all_fista_solutions]), axis=0)
        / np.sqrt(loops)
        for size in sizes
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for array, array_ste, name, style, color in zip(
        residuals_fista_mean + residuals_flpdap_mean + [lpdap_residuals_mean],
        residuals_fista_ste + residuals_flpdap_ste + [lpdap_residuals_ste],
        names,
        styles,
        colors,
    ):
        ax.loglog(np.arange(len(array)), array, style, label=name, color=color)
        ax.fill(
            np.hstack(
                (
                    np.arange(len(array)),
                    np.arange(len(array))[::-1],
                )
            ),
            np.hstack(
                (
                    np.maximum(
                        np.array(array) - np.array(array_ste),
                        1e-13,
                    ),
                    np.array(array)[::-1] + np.array(array_ste)[::-1],
                )
            ),
            color,
            alpha=0.3,
        )
    plt.ylabel("Objective residual")
    plt.xlabel("Iterations")
    plt.ylim(1e-10, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "grid_res_iter.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


if __name__ == "__main__":
    experiment()
