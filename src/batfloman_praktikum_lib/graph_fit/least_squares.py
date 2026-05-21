from typing import Optional, Union, Callable, Type, Mapping, Any
from inspect import isclass
import numpy as np
from scipy.optimize import curve_fit

from batfloman_praktikum_lib.graph_fit.models.fitModel import FitModel
from .helper import extract_vals_and_errors
from ..graph.plotNScatter import filter_nan_values
from .fitResult import generate_fit_result, FitResult
from .fixed_params import (
    build_fixed_param_binding,
    order_free_initial_guess,
    rebuild_full_fit_result,
)

from .init_params.order_init_params import InitalParamGuess

from .user_warnings import warn_user_no_y_errors_least_squares, warn_user_x_errors_least_squares

def generic_fit(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    y_err=None,
    *,
    initial_guess: Optional[InitalParamGuess] = None,
    fixed_params: Mapping[str, Any] | None = None,
    param_names = None,
    ignore_warning_x_errors: bool = False,
    ignore_warning_y_errors: bool = False,
) -> FitResult:
    if isclass(model) and issubclass(model, FitModel):
        if not param_names:
            param_names = model.get_param_names()
        model = model.model

    x_data, y_data = filter_nan_values(x_data, y_data, warn_filter_nan=True)

    warn_user_x_errors_least_squares(x_data, ignore_warning_x_errors)
    warn_user_no_y_errors_least_squares(y_data, y_err, ignore_warning_y_errors)

    y_data, y_err = extract_vals_and_errors(y_data, y_err)
    x_data, _     = extract_vals_and_errors(x_data, None)

    binding = build_fixed_param_binding(model, fixed_params=fixed_params)
    fit_model = binding.wrap_model() if binding.fixed_params else model
    fit_param_names = binding.free_param_names if binding.fixed_params else param_names
    initial_guess = order_free_initial_guess(
        model,
        initial_guess,
        binding=binding,
    )

    if binding.fixed_params and not binding.free_param_names:
        chi_squared_red = _calc_chi_squared(fit_model, x_data, y_data, y_err, [])
        return rebuild_full_fit_result(
            binding=binding,
            free_values=[],
            free_errors=[],
            cov=np.zeros((0, 0), dtype=float),
            quality=chi_squared_red,
            method="least squares",
        )

    # Perform the curve fit
    popt, pcov = curve_fit(
        fit_model,
        x_data,
        y_data,
        sigma=y_err,
        absolute_sigma=True,
        p0=initial_guess,
    )

    # Uncertainties from covariance matrix (sqrt of diagonal elements)
    perr = np.sqrt(np.diag(pcov))

    chi_squared_red = _calc_chi_squared(fit_model, x_data, y_data, y_err, popt)
    
    if binding.fixed_params:
        return rebuild_full_fit_result(
            binding=binding,
            free_values=popt,
            free_errors=perr,
            cov=pcov,
            quality=chi_squared_red,
            method="least squares",
        )

    return generate_fit_result(model, popt, perr, pcov, param_names=fit_param_names, quality=chi_squared_red, method="least squares");

def _calc_chi_squared(model, x_data, y_data, yerr, popt):
    residuals = (y_data - model(x_data, *popt)) / yerr
    chi_squared = np.sum(residuals**2)

    # Compute reduced chi-squared
    dof = len(y_data) - len(popt)  # Degrees of freedom = N - k
    chi_squared_red = chi_squared / dof if dof > 0 else np.nan  # Avoid division by zero

    return chi_squared_red
