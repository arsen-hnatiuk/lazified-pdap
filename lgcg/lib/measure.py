import numpy as np
from typing import Callable


class Measure:
    # An implementation of a class that represents measures on Omega
    def __init__(
        self,
        support: np.ndarray = np.array([]),
        coefficients: np.ndarray = np.array([]),
    ) -> None:
        self.support = np.unique(support, axis=0)
        self.coefficients = coefficients
        assert len(self.support) == len(
            self.coefficients
        ), "The support and coefficients must have the same length"

    def multiply(self, constant: float) -> None:
        self.coefficients *= constant

    def add_zero_support(self, support_plus: np.ndarray) -> None:
        for point in support_plus:
            if point not in self.support:
                if len(self.support.shape) == 1:
                    self.support = np.append(self.support, point)
                else:
                    self.support = np.vstack([self.support, point])
                self.coefficients = np.append(self.coefficients, 0)

    def duality_pairing(self, fct: Callable) -> float:
        return sum([c * fct(x) for x, c in zip([self.support, self.coefficients])])


def add_measures(u: Measure, v: Measure) -> Measure:
    new = Measure(support=u.support, coefficients=u.coefficients)
    for x, c in zip(v.support, v.coefficients):
        for i, pos in enumerate(new.support):
            if pos == x:
                new.coefficients[i] += c
                break
        # new support point
        new.add_zero_support(np.array([x]))
    return new
