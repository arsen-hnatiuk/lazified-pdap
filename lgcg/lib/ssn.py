# An implementation of the semismooth Newton method following https://epubs.siam.org/doi/epdf/10.1137/120892167

import numpy as np
from lib.default_values import *


class SSN:
    def __init__(self, K: np.ndarray, alpha: float, y_true: np.ndarray) -> None:
        self.K = K
        self.y_true = y_true
        self.alpha = alpha
        self.g = get_default_g(self.alpha)
        self.f = get_default_f(self.K, self.y_true)
        self.j = lambda u: self.f(u) + self.g(u)
        self.L = 1
        self.norm_K_star = max(
            [np.linalg.norm(row) for row in np.transpose(self.K)]
        )  # the 2,inf norm of the transpose of K
        self.gamma = 1
