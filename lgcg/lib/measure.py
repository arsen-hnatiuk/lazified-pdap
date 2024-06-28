import numpy as np
from typing import Callable


class Measure:
    # An implementation of a class that represents finitely supported measures on Omega
    def __init__(
        self,
        support: np.ndarray = np.array([]),
        coefficients: np.ndarray = np.array([]),
    ) -> None:
        self.support = np.unique(np.array(support), axis=0)
        self.coefficients = np.array(coefficients)
        assert len(self.support) == len(
            self.coefficients
        ), "The support and coefficients must have the same length"

    def add_zero_support(self, support_plus: np.ndarray) -> None:
        # Given an array of points, add them to the support with coefficient 0
        for point in support_plus:
            if point not in self.support:
                if len(self.support.shape) == 1:
                    self.support = np.array([point])
                else:
                    self.support = np.vstack([self.support, point])
                self.coefficients = np.append(self.coefficients, 0)

    def duality_pairing(self, fct: Callable) -> float:
        # Compute the duality pairing of the measure with a function defined on Omega
        return sum([c * fct(x) for x, c in zip(self.support, self.coefficients)])

    def __add__(self, other):
        # Add two measures
        new = Measure(
            support=self.support.copy(), coefficients=self.coefficients.copy()
        )
        for x, c in zip(other.support, other.coefficients):
            changed = False
            for i, pos in enumerate(new.support):
                if np.array_equal(pos, x):
                    new.coefficients[i] += c
                    changed = True
                    break
            # new support point
            if not changed:
                new.add_zero_support(np.array([x]))
                new.coefficients[-1] = c
        return new

    def __mul__(self, other):
        # Multiply a measure by a scalar
        new = Measure(
            support=self.support.copy(), coefficients=self.coefficients.copy() * other
        )
        return new
