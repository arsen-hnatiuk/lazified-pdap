import numpy as np
import sys
import logging
import matplotlib.pyplot as plt
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))
from lib.measure import Measure
from lazified_pdap import LazifiedPDAP

results_dir = Path("results/signal_processing")
results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

observation_resolution = 120
Omega = np.array([[0.0, observation_resolution // 2]])
alpha = 1e-1
true_sources = np.array([[3.125], [7], [np.sqrt(179)]])
true_weights = np.array([-1, 0.7, 0.5])

observations = np.arange(0, 1, 1 / observation_resolution)


def kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    columns = []
    for t in observations:
        column = np.sin(2 * np.pi * x * t).flatten()
        columns.append(column)
    result = np.transpose(
        np.array(columns), axes=(1, 0)
    )  # shape=(len(x), len(observations))
    return result


def grad_kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    gradients = []
    for t in observations:
        factor = 2 * np.pi * t
        column = np.cos(2 * np.pi * x * t)  # (len(x), Omega.shape[0])
        gradient = factor * column
        gradients.append(gradient)
    result = np.transpose(
        np.array(gradients), axes=(1, 0, 2)
    )  # The Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0])
    return result


def hess_kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    hessians = []
    for t in observations:
        factor = -4 * np.pi**2 * t**2
        column = np.sin(2 * np.pi * x * t).reshape(
            -1, 1, 1
        )  # (len(x), Omega.shape[0], Omega.shape[0])
        hessian = factor * column
        hessians.append(hessian)
    result = np.transpose(
        np.array(hessians), axes=(1, 0, 2, 3)
    )  # The derivative of the Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0], Omega.shape[0])
    return result


u_hat = Measure(support=true_sources, coefficients=true_weights)
target = u_hat.duality_pairing(kernel)

g = lambda u: alpha * np.linalg.norm(u, ord=1)
f = lambda u: 0.5 * np.linalg.norm(u.duality_pairing(kernel) - target) ** 2


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


a = np.arange(Omega[0][0], Omega[0][1], 0.1).reshape(-1, 1)
vals = np.linalg.norm(kernel(a), axis=1)
norm_kernel = max(vals)

a = np.arange(Omega[0][0], Omega[0][1], 0.1).reshape(-1, 1)
vals = np.linalg.norm(grad_kernel(a), axis=(1, 2))
norm_kernel1 = max(norm_kernel, max(vals))
norm_kernel1

gamma = 1
theta = 1e-1
sigma = 5e-2
m = 1e-3
bar_m = 1e-1
L = 1
L_H = max(np.abs(true_weights))
R = 1e-1


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


def experiment():
    exp = LazifiedPDAP(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        p=p,
        grad_P=grad_P,
        hess_P=hess_P,
        norm_kappa=norm_kernel,
        norm_kappa1=norm_kernel,
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
        tol=1e-14
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

    optimum = objective_values_pdap[-1]
    residuals_pdap = objective_values_pdap - optimum
    residuals_nlgcg = objective_values_nlgcg - optimum
    residuals_lpdap = objective_values_lpdap - optimum
    logging.info("-------------------------------------------------------------------")

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot Signal
    plt.plot(observations, target)
    plt.xlabel("Time")
    plt.savefig(results_dir / "input_signal.png", bbox_inches="tight")
    plt.close()

    # Plot dual variable
    u_tilde = u_nlgcg  # Newton solution
    a = np.arange(Omega[0][0], Omega[0][1], 0.005)
    p_u = p(u_tilde)
    vals = np.abs(p_u(a))
    plt.plot(a, vals)
    for i, pos in enumerate(u_tilde.support):
        if not i:
            plt.axvline(x=pos, linestyle="--", c="r", label="Optimal and true support")
        else:
            plt.axvline(x=pos, linestyle="--", c="r")
    plt.xlabel("Frequency")
    plt.ylim(0, 0.11)
    plt.xlim(0, 20)
    plt.legend()
    plt.savefig(results_dir / "signal_dual_certificate.png", bbox_inches="tight")
    plt.close()

    # Plot iterate dual variable
    u_tilde = intermediate_u_lpdap  # LPDAP iterate after 30 ierations
    a = np.arange(Omega[0][0], Omega[0][1], 0.005)
    p_u = p(u_tilde)
    vals = np.abs(p_u(a))
    plt.plot(a, vals)
    for i, pos in enumerate(u_tilde.support):
        if not i:
            plt.plot(pos, 0.001, "P", c="r", label="Iterate support", markersize=10)
        else:
            plt.plot(pos, 0.001, "P", c="r", markersize=10)
    plt.xlabel("Frequency")
    plt.ylim(0, 0.13)
    plt.xlim(0, 20)
    plt.legend()
    plt.savefig(results_dir / "iterate_dual.png", bbox_inches="tight")
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
    plt.ylim(1e-12, 55)
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
    styles = ["-", "--", "-."]
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
    plt.ylim(1e-12, 55)
    plt.legend()
    plt.savefig(results_dir / "res_time.png", bbox_inches="tight")
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
