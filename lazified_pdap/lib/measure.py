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
        support, index = np.unique(np.array(support), axis=0, return_index=True)
        coefficients = np.array(coefficients)[index].astype(float)
        non_zero_index = np.where(coefficients != 0)[0]
        self.support = support[non_zero_index]
        self.coefficients = coefficients[non_zero_index]
        assert len(self.support) == len(
            self.coefficients
        ), "The support and coefficients must have the same length"

    def add_zero_support(self, support_plus: np.ndarray, checked: bool = False) -> None:
        # Given an array of points, add them to the support with coefficient 0
        for point in support_plus:
            if len(self.support.shape) == 1:  # No support
                self.support = np.array([point])
                self.coefficients = np.append(self.coefficients, 0)
            elif checked or not np.any(self.support == point, axis=1):
                self.support = np.vstack([self.support, point])
                self.coefficients = np.append(self.coefficients, 0)

    def duality_pairing(self, fct: Union[np.ndarray, Callable]) -> float:
        # Compute the duality pairing of the measure with a function defined on Omega
        if not len(self.support):
            return 0
        if type(fct) == np.ndarray:
            values = fct[self.support.flatten()]
        else:
            values = fct(self.support.copy())
        if len(values.shape) > 1:
            values = values.T
            result = values @ self.coefficients
            result = result.flatten()
        else:
            result = values @ self.coefficients
        return result

    def __add__(self, other: "Measure") -> "Measure":
        # Add two measures
        if not isinstance(other, Measure):
            return NotImplemented
        if not len(self.support):
            return other
        if not len(other.support):
            return self
        new = Measure(
            support=self.support.copy(), coefficients=self.coefficients.copy()
        )
        for x, c in zip(other.support, other.coefficients):
            changed = False
            matches = np.all(new.support == x, axis=1)
            if np.any(matches):
                idx = np.where(matches)[0][0]
                new.coefficients[idx] += c
                changed = True
            if not changed:
                new.add_zero_support(np.array([x]), checked=True)
                new.coefficients[-1] = c
        return new

    def __mul__(self, other: float) -> "Measure":
        # Multiply a measure by a scalar
        new = Measure(
            support=self.support.copy(), coefficients=self.coefficients.copy() * other
        )
        return new

    def __str__(self) -> str:
        return (
            f"Measure with support {self.support} and coefficients {self.coefficients}"
        )
