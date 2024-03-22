import numpy as np
from typing import Union, Callable


def get_default_K(y: np.ndarray) -> np.ndarray:
    size = y.shape[0]
    return np.random.rand(size, 2 * size)


def get_default_f(K: np.ndarray, y: np.ndarray) -> Callable:
    # 0.5\Vert Ku-y\Vert^2
    return lambda u: 0.5 * np.linalg.norm(np.matmul(K, u) - y) ** 2


def get_default_p(K: np.ndarray, y: np.ndarray) -> Callable:
    # (K_*(Ku-y)*e_x)
    return lambda u: np.matmul(np.transpose(K), np.matmul(K, u) - y)


def get_default_g(alpha: float) -> Callable:
    return lambda u: alpha * np.linalg.norm(u, ord=1)


def get_default_Omega(K: np.ndarray) -> np.ndarray:
    return np.array(range(K.shape[1]))
