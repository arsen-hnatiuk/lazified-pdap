import numpy as np
from typing import Callable


def get_default_f(K: np.ndarray, y: np.ndarray) -> Callable:
    # 0.5\Vert Ku-y\Vert^2
    return lambda u: 0.5 * np.linalg.norm(np.matmul(K, u) - y) ** 2


def get_default_p(K: np.ndarray, y: np.ndarray) -> Callable:
    # K_*(Ku-y)
    return lambda u: -np.matmul(np.transpose(K), np.matmul(K, u) - y)


def get_default_g(alpha: float) -> Callable:
    return lambda u: alpha * np.linalg.norm(u, ord=1)


def get_grad_j(
    k: Callable, grad_k: Callable, alpha: float, target: np.ndarray
) -> Callable:
    def grad_j(positions: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        to_return = []
        grad_F = (
            np.sum(
                np.array([c * k(x) for x, c in zip(positions, coefs)]),
                axis=0,
            )
            - target
        )
        for ind, x in enumerate(positions):
            # \nabla_x
            array = coefs[ind] * np.matmul(grad_k(x).T, grad_F)
            to_return += array.tolist()
        for ind, c in enumerate(coefs):
            # \nabla_u
            to_return.append(np.dot(k(x), grad_F) + alpha * np.sing(c))
        return np.array(to_return)

    return grad_j


def get_hess_j(
    k: Callable, grad_k: Callable, hess_k: Callable, alpha: float, target: np.ndarray
) -> Callable:
    def hess_j(positions: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        return
