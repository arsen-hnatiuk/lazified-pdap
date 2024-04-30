import numpy as np
from typing import Callable


class LSI:
    # Implementation of the local support improver
    def __init__(
        self, active_set: np.ndarray, p: Callable, tol: float, gap: float
    ) -> None:
        self.active_set = active_set
        self.p = p
        self.tol = tol
        self.gap = gap
