import warnings
from collections.abc import Callable
import numpy as np

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase


def _is_sequence_x(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray)) and np.ndim(x) > 0


def _should_scalar_fallback(exc: Exception) -> bool:
    message = str(exc)
    return (
        "truth value of an array" in message
        or "ambiguous" in message and "truth value" in message
    )


def evaluate_model(model: Callable, x, *params):
    if not _is_sequence_x(x):
        return model(x, *params)

    try:
        return model(x, *params)
    except ValueError as exc:
        if not _should_scalar_fallback(exc):
            raise
        return np.asarray([model(x_val, *params) for x_val in x])


def extract_vals_and_errors(vals, errs):
    if all(hasattr(val, "value") and hasattr(val, "error") for val in vals):
        extracted_errs = np.array([val.error for val in vals])
        vals = np.array([val.value for val in vals])
        if errs is None:
            errs = extracted_errs

    vals = np.asarray(vals, dtype=float)
    if errs is not None:
        errs = np.asarray(errs, dtype=float)

    if errs is None or np.all(errs == 0):
        errs = np.ones_like(vals, dtype=float)  # Assume equal weights if no error provided

    if np.any(errs <= 0):
        raise ValueError("Error values must be positive and non-zero for meaningful fitting.")

    return vals, errs
