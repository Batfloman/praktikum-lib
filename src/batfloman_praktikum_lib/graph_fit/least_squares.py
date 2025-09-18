from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
import numpy as np
from scipy.optimize import curve_fit

from .fitResult import generate_fit_result, FitResult

def generic_fit(model, x_data, y_data, yerr=None, initial_guess=None, param_names = None) -> FitResult:
    if all(hasattr(y, "value") and hasattr(y, "error") for y in y_data):
        # Extract values and errors
        yerr = np.array([y.error for y in y_data])
        y_data = np.array([y.value for y in y_data])

    if initial_guess is not None:
        initial_guess = [guess.value if isinstance(guess, MeasurementBase) else guess for guess in initial_guess]

    if yerr is None or np.all(yerr == 0):
        yerr = np.ones_like(y_data)  # Assume equal weights if no error provided
    if np.any(yerr <= 0):
        raise ValueError("Error values in 'yerr' must be positive and non-zero for meaningful fitting.")

    # Perform the curve fit
    popt, pcov = curve_fit(model, x_data, y_data, sigma=yerr, absolute_sigma=True, p0=initial_guess)

    # Uncertainties from covariance matrix (sqrt of diagonal elements)
    perr = np.sqrt(np.diag(pcov))

    chi_squared_red = _calc_chi_squared(model, x_data, y_data, yerr, popt)
    
    # return popt, perr
    return generate_fit_result(model, popt, perr, param_names, quality=chi_squared_red);

def _calc_chi_squared(model, x_data, y_data, yerr, popt):
    residuals = (y_data - model(x_data, *popt)) / yerr
    chi_squared = np.sum(residuals**2)

    # Compute reduced chi-squared
    dof = len(y_data) - len(popt)  # Degrees of freedom = N - k
    chi_squared_red = chi_squared / dof if dof > 0 else np.nan  # Avoid division by zero

    return chi_squared_red
