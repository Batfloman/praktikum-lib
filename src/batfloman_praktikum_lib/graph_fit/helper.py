import warnings
import numpy as np

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase

def extract_vals_and_errors(vals, errs):
    if all(hasattr(val, "value") and hasattr(val, "error") for val in vals):
        # Extract values and errors
        errs = np.array([val.error for val in vals])
        vals = np.array([val.value for val in vals])

    if errs is None or np.all(errs == 0):
        errs = np.ones_like(vals)  # Assume equal weights if no error provided

    if np.any(errs <= 0):
        raise ValueError("Error values must be positive and non-zero for meaningful fitting.")

    return vals, errs
