from typing import Union, Callable, Type
from inspect import isclass
import numpy as np
from scipy.optimize import curve_fit
import warnings

from batfloman_praktikum_lib.graph_fit.models.fitModel import FitModel
from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from .helper import extract_vals_and_errors
from ..graph.plotNScatter import filter_nan_values
from .fitResult import generate_fit_result, FitResult
from .find_initial_parameters import order_initial_params

def generic_fit(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    y_err=None,
    initial_guess=None,
    param_names = None
) -> FitResult:
    if isclass(model) and issubclass(model, FitModel):
        if not param_names:
            param_names = model.get_param_names()
        model = model.model

    if any(isinstance(x, MeasurementBase) and x.error is not None for x in x_data):
        warnings.warn(
            "\nx-value uncertainties were detected but are ignored by least-squares fitting. "
            "\nUse ODR if x-errors should be included.",
            UserWarning
        )

    y_data, y_err = extract_vals_and_errors(y_data, y_err)
    x_data, _     = extract_vals_and_errors(x_data, None)

    if initial_guess is not None:
        initial_guess = order_initial_params(model, initial_guess);
        initial_guess = [guess.value if isinstance(guess, MeasurementBase) else guess for guess in initial_guess]

    # Perform the curve fit
    popt, pcov = curve_fit(model, x_data, y_data, sigma=y_err, absolute_sigma=True, p0=initial_guess)

    # Uncertainties from covariance matrix (sqrt of diagonal elements)
    perr = np.sqrt(np.diag(pcov))

    chi_squared_red = _calc_chi_squared(model, x_data, y_data, y_err, popt)
    
    # return popt, perr
    return generate_fit_result(model, popt, perr, pcov, param_names=param_names, quality=chi_squared_red);

def _calc_chi_squared(model, x_data, y_data, yerr, popt):
    residuals = (y_data - model(x_data, *popt)) / yerr
    chi_squared = np.sum(residuals**2)

    # Compute reduced chi-squared
    dof = len(y_data) - len(popt)  # Degrees of freedom = N - k
    chi_squared_red = chi_squared / dof if dof > 0 else np.nan  # Avoid division by zero

    return chi_squared_red
