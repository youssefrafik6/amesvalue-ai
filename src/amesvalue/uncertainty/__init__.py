"""Finite-sample split conformal absolute-error intervals."""

import math

import numpy as np


def conformal_radius(y, prediction, coverage: float = 0.9) -> float:
    """Return the ceil((n+1)*coverage)-th ordered calibration error."""
    observed = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if observed.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("Calibration values must be aligned one-dimensional arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(observed - predicted)
    if not np.isfinite(errors).all():
        raise ValueError("Calibration errors must be finite")
    if not 0 < coverage < 1:
        raise ValueError("Nominal coverage must be between zero and one")
    rank = math.ceil((len(errors) + 1) * coverage)
    if rank > len(errors):
        raise ValueError("Insufficient calibration observations for nominal coverage")
    return float(np.partition(errors, rank - 1)[rank - 1])


def interval(prediction, radius):
    """Apply the physical lower bound of zero dollars."""
    values = np.asarray(prediction)
    return np.maximum(values - radius, 0), np.maximum(values + radius, 0)
