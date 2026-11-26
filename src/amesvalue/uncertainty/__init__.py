"""Finite-sample split conformal absolute-error intervals."""

import math

import numpy as np


def conformal_radius(y, prediction, coverage: float = 0.9) -> float:
    """Return the ceil((n+1)*coverage)-th ordered calibration error."""
    errors = np.abs(np.asarray(y) - np.asarray(prediction))
    rank = math.ceil((len(errors) + 1) * coverage)
    if not 0 < coverage < 1 or rank > len(errors):
        raise ValueError("Insufficient calibration observations for nominal coverage")
    return float(np.partition(errors, rank - 1)[rank - 1])


def interval(prediction, radius):
    """Apply the physical lower bound of zero dollars."""
    values = np.asarray(prediction)
    return np.maximum(values - radius, 0), np.maximum(values + radius, 0)
