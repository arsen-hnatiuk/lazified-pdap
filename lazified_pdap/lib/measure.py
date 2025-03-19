import numpy as np
from typing import Callable, Union
import logging


class Measure:
    # An implementation of a class that represents finitely supported measures on Omega
    def __init__(
        self,
        support: np.ndarray = np.array([]),
        coefficients: np.ndarray = np.array([]),
    ) -> None:
        self.support, index = np.unique(np.array(support), axis=0, return_index=True)
        self.coefficients = np.array(coefficients)[index].astype(float)
        assert len(self.support) == len(
            self.coefficients
        ), "The support and coefficients must have the same length"

    def add_zero_support(self, support_plus: np.ndarray) -> None:
        # Given an array of points, add them to the support with coefficient 0
        for point in support_plus:
            if len(self.support.shape) == 1:  # No support
                self.support = np.array([point])
                self.coefficients = np.append(self.coefficients, 0)
            elif all(
                [
                    not np.array_equal(point, support_point)
                    for support_point in self.support
                ]
            ):
                self.support = np.vstack([self.support, point])
                self.coefficients = np.append(self.coefficients, 0)

    def duality_pairing(self, fct_in: Union[np.ndarray, Callable]) -> float:
        # Compute the duality pairing of the measure with a function defined on Omega
        if not len(self.support):
            return 0
        if type(fct_in) == np.ndarray:
            fct = lambda x: fct_in[x.flatten()]  # turn matrix into Callable
        else:
            fct = fct_in
        values = fct(self.support.copy())
        if len(values.shape) > 1:
            values = values.T
            result = values @ self.coefficients
            result = result.flatten()
        else:
            result = values @ self.coefficients
        return result

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

    def __str__(self):
        return (
            f"Measure with support {self.support} and coefficients {self.coefficients}"
        )
