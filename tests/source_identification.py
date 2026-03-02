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

results_dir = Path("results/source_identification")
results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
Omega_size = Omega[0][1] - Omega[0][0]
alpha = 1e-1
observation_resolution = 4
std_factor = 0.1
true_sources = np.array([[0.28, 0.71], [0.51, 0.27], [0.71, 0.53]])
true_weights = np.array([1, -0.7, 0.8])

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
        gradient = diff * exponentiated.reshape(-1, 1)  # shape=(len(x),Omega.shape[0])
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
            hesse_matrix[i * step : (i + 1) * step, j * step : (j + 1) * step] = block
            hesse_matrix[j * step : (j + 1) * step, i * step : (i + 1) * step] = block.T
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

gamma = 1
theta = 1e-1
sigma = 2e-3
m = 1e-3
bar_m = 1e-1
L = 1
L_H = max(np.abs(true_weights))
R = 1e-2


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


def experiment():
    exp = LazifiedPDAP(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        p=p,
        grad_P=grad_P,
        hess_P=hess_P,
        norm_kernel=norm_kernel,
        norm_kernel1=norm_kernel,
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
    u_pdap, P_values_pdap, times_pdap, supports_pdap, objective_values_pdap = exp.pdap(
        tol=1e-12
    )
    logging.info("-------------------------------------------------------------------")

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
    ) = exp.newton(tol=1e-12, damped=False)
    intervals_nlgcg = compute_intervals(inner_loop_nlgcg)
    logging.info(f"Lazy LGCG steps: {lgcg_lazy_nlgcg}")
    logging.info(f"Total LGCG steps: {lgcg_total_nlgcg}")
    logging.info("-------------------------------------------------------------------")

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
    ) = exp.lpdap(tol=1e-12)
    logging.info("-------------------------------------------------------------------")

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
        u_fista, objective_values_fista, times_fista = fista_exp.solve(max_time=60)
        fista_solutions[size] = (u_fista, objective_values_fista - optimum, times_fista)
        logging.info(
            "-------------------------------------------------------------------"
        )

        logging.info(
            f"Solving with finite LPDAP on uniform grid of size {int(size**2)}"
        )
        flpdap_exp = LazifiedPDAPFinite(
            K_transpose=K_transpose, alpha=alpha, target=target
        )
        u_flpdap, objective_values_flpdap, times_flpdap = flpdap_exp.solve(tol=1e-10)
        flpdap_solutions[size] = (
            u_flpdap,
            objective_values_flpdap - optimum,
            times_flpdap,
        )
        logging.info(
            "-------------------------------------------------------------------"
        )

    residuals_pdap = objective_values_pdap - optimum
    residuals_nlgcg = objective_values_nlgcg - optimum
    residuals_lpdap = objective_values_lpdap - optimum

    for size in sizes:
        u_flpdap, residuals_flpdap, times_flpdap = flpdap_solutions[size]
        logging.info(
            f"Finite LPDAP on grid of size {size}: absolute residual {residuals_flpdap[-1]:.12E}, time {times_flpdap[-1]:.3f}s"
        )
        u_sklearn, residual_sklearn, time_sklearn = sklearn_solutions_default[size]
        logging.info(
            f"Scikit-learn default on grid of size {size}: absolute residual {residual_sklearn:.12E}, grid residual {residual_sklearn - residuals_flpdap[-1]:.12E}, time {time_sklearn:.3f}s"
        )
        u_sklearn, residual_sklearn, time_sklearn = sklearn_solutions_custom[size]
        logging.info(
            f"Scikit-learn custom on grid of size {size}: absolute residual {residual_sklearn:.12E}, grid residual {residual_sklearn - residuals_flpdap[-1]:.12E}, time {time_sklearn:.3f}s"
        )

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot cluster
    clustered_points = [
        [0.28322656, 0.7143313],
        [0.28322701, 0.71433113],
        [0.28322748, 0.71433135],
        [0.28322768, 0.71433104],
    ]
    clustered_weights = [0.21149885, 0.1, 0.75096038, 0.07]
    true_point = [0.28322727, 0.71433132]
    true_weight = 0.99569143
    plt.plot(0, 0, "o", c="b", markersize=10, label="Clustered support")
    # Point for legend
    plt.plot(0, 0, "o", c="r", fillstyle="none", markersize=10, label="Optimal support")
    # Point for legend
    for x, c in zip(clustered_points, clustered_weights):
        plt.plot(x[0], x[1], "o", c="b", markersize=25 * c)
    plt.plot(
        true_point[0],
        true_point[1],
        "o",
        fillstyle="none",
        c="r",
        markersize=25 * true_weight,
    )
    plt.ylim(7.143313e-1 - 7e-7, 7.143313e-1 + 7e-7)
    plt.xlim(2.832272e-1 - 7e-7, 2.832272e-1 + 7e-7)
    plt.legend(fontsize=10)
    plt.savefig(results_dir / "clustering.png", bbox_inches="tight")
    plt.close()

    # Plot dual variable
    u_tilde = u_nlgcg
    p_u = p(u_tilde)
    P = lambda x: np.abs(p_u(x))
    a = np.arange(0, 1, 0.01)
    B, D = np.meshgrid(a, a)
    vals = np.array(
        [P(np.array([x_1, x_2])) for x_1, x_2 in zip(B.flatten(), D.flatten())]
    ).reshape((100, 100))
    plt.contourf(B, D, vals, levels=100)
    plt.colorbar()
    for i, x in enumerate(true_sources):
        if i:
            plt.plot([x[0]], [x[1]], "P", c="r", markersize=10)
        else:
            plt.plot([x[0]], [x[1]], "P", c="r", markersize=10, label="True sources")
    for i, x in enumerate(u_tilde.support):
        if i:
            plt.plot([x[0]], [x[1]], "o", c="b")
        else:
            plt.plot([x[0]], [x[1]], "o", c="b", label="Optimal support")
    plt.legend()
    plt.savefig(results_dir / "optimal_dual_certificate.png", bbox_inches="tight")
    plt.close()

    # Plot residuals
    names = ["PDAP", "LPDAP", "NLGCG"]
    styles = ["-", "--", "-."]
    plt.figure(figsize=(11.25, 5))
    for array, name, style in zip(
        [residuals_pdap, residuals_lpdap, residuals_nlgcg], names, styles
    ):
        plt.semilogy(np.array(range(len(array))), array, style, label=name)
    plt.ylabel("Objective residual")
    plt.xlabel("Total iterations")
    plt.ylim(1e-12, 30)
    plt.legend()
    plt.savefig(results_dir / "res_iter.png", bbox_inches="tight")
    plt.close()

    # Plot lazy thresholds
    names = ["LPDAP", "NLGCG"]
    styles = ["--", "-."]
    colors = ["orange", "green"]
    plt.figure(figsize=(11.25, 5))
    for array, name, style, color in zip(
        [epsilons_lpdap, epsilons_nlgcg], names, styles, colors
    ):
        plt.semilogy(np.array(range(len(array))), array, style, label=name, color=color)
    plt.ylabel("Lazy threshold")
    plt.xlabel("Total iterations")
    plt.ylim(1e-12, 0.1)
    plt.legend()
    plt.savefig(results_dir / "eps_iter.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    names = ["PDAP", "LPDAP", "NLGCG"]
    styles = ["-", "--", "-."]
    plt.figure(figsize=(11.25, 5))
    for array, name, style in zip(
        [supports_pdap, supports_lpdap, supports_nlgcg], names, styles
    ):
        plt.plot(np.array(range(len(array))), array, style, label=name)
    plt.ylabel("Support size")
    plt.xlabel("Total iterations")
    plt.legend()
    plt.savefig(results_dir / "support_iter.png", bbox_inches="tight")
    plt.close()

    # Plot residuals in time
    names = ["PDAP", "LPDAP", "NLGCG"]
    styles = ["-", "--", "-.", ":"]
    plt.figure(figsize=(11.25, 5))
    for domain, array, name, style in zip(
        [times_pdap, times_lpdap, times_nlgcg],
        [residuals_pdap, residuals_lpdap, residuals_nlgcg],
        names,
        styles,
    ):
        plt.semilogy(domain, array, style, label=name)
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-12, 30)
    plt.legend()
    plt.savefig(results_dir / "res_time.png", bbox_inches="tight")
    plt.close()

    # Plot FLPDAP vs FISTA residuals on grid in time
    names = (
        [f"FISTA, mesh {Omega_size/size}" for size in sizes]
        + [f"Finite LPDPA, mesh {Omega_size/size}" for size in sizes]
        + [f"LPDAP, mesh {0}"]
    )
    colors = ["blue"] * len(sizes) + ["green"] * len(sizes) + ["black"]
    styles = ["-", "--", "-."] + ["-", "--", "-."] + ["-"]
    residuals_flpdap = [res for _, res, _ in flpdap_solutions.values()]
    times_flpdap = [tim for _, _, tim in flpdap_solutions.values()]
    residuals_fista = [res for _, res, _ in fista_solutions.values()]
    times_fista = [tim for _, _, tim in fista_solutions.values()]
    plt.figure(figsize=(11.25, 5))
    for domain, array, name, style, color in zip(
        times_fista + times_flpdap + [times_lpdap],
        residuals_fista + residuals_flpdap + [residuals_lpdap],
        names,
        styles,
        colors,
    ):
        plt.loglog(domain, array, style, label=name, color=color)
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-10, 100)
    plt.legend()
    plt.savefig(results_dir / "grid_res_time.png", bbox_inches="tight")
    plt.close()

    # Plot FLPDAP vs FISTA residuals on grid in iters
    names = (
        [f"FISTA, mesh {Omega_size/size}" for size in sizes]
        + [f"Finite LPDPA, mesh {Omega_size/size}" for size in sizes]
        + [f"LPDAP, mesh {0}"]
    )
    colors = ["blue"] * len(sizes) + ["green"] * len(sizes) + ["black"]
    styles = ["-", "--", "-."] + ["-", "--", "-."] + ["-"]
    residuals_flpdap = [res for _, res, _ in flpdap_solutions.values()]
    residuals_fista = [res for _, res, _ in fista_solutions.values()]
    plt.figure(figsize=(11.25, 5))
    for domain, array, name, style, color in zip(
        [
            np.arange(len(ar))
            for ar in residuals_fista + residuals_flpdap + [residuals_lpdap]
        ],
        residuals_fista + residuals_flpdap + [residuals_lpdap],
        names,
        styles,
        colors,
    ):
        plt.loglog(domain, array, style, label=name, color=color)
    plt.ylabel("Objective residual")
    plt.xlabel("Iterations")
    plt.ylim(1e-10, 100)
    # plt.xlim(0, 100)
    plt.legend()
    plt.savefig(results_dir / "grid_res_iter.png", bbox_inches="tight")
    plt.close()

    # Plot inner loop
    plt.figure(figsize=(11.25, 5))
    plt.semilogy(
        np.array(range(len(residuals_nlgcg))),
        residuals_nlgcg,
        linestyle="-.",
        color="green",
        label="Objective residual",
    )
    plt.semilogy(
        np.array(range(len(epsilons_nlgcg))),
        epsilons_nlgcg,
        linestyle="-",
        color="red",
        label="Lazy threshold",
    )
    for interval in intervals_nlgcg:
        plt.fill_between(interval, 0, 60, hatch="/", color="gray", alpha=0.2)
    plt.fill_between(
        interval, -10, -9, hatch="/", color="gray", alpha=0.2, label="Inner loop"
    )
    plt.ylim(1e-14, 60)
    plt.legend()
    plt.xlabel("Total iterations")
    plt.savefig(results_dir / "local_routine.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


if __name__ == "__main__":
    experiment()
