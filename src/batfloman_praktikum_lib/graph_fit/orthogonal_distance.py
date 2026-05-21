from typing import Optional, Type, Callable, Union, Literal, Mapping, Any
from inspect import isclass
import numpy as np
import traceback
from scipy.odr import ODR, Model, RealData

from batfloman_praktikum_lib.io.termColors import bcolors

from ..graph.plotNScatter import filter_nan_values
from .helper import extract_vals_and_errors
from .fitResult import generate_fit_result, FitResult
from .fixed_params import (
    build_fixed_param_binding,
    order_free_initial_guess,
    rebuild_full_fit_result,
)
from .models.fitModel import FitModel
from .init_params.order_init_params import InitalParamGuess

def _warn_user_no_errors(y_data, y_err, ignore_y_errors: bool, coord: Literal["x", "y"]):
    if ignore_y_errors or (y_err is not None):
        return

    has_errors = False
    try:
        # z.B. iterierbare Messwerte mit .error
        has_errors = all(hasattr(y, "error") and y.error is not None for y in y_data)
    except TypeError:
        # z.B. Skalar mit .error
        has_errors = hasattr(y_data, "error") and y_data.error is not None

    if not has_errors:
        stack = traceback.extract_stack()
        frame = stack[-2]

        print(f"{bcolors.WARNING}Warning: no {coord}-value uncertainties were detected, using equal weights !{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}{bcolors.BOLD} At Line {frame.lineno}{bcolors.ENDC}: `{frame.line}`")
        print(f"\tin {frame.filename}:{frame.lineno}:0")
        print(f"{bcolors.WARNING} - Call with `ignore_warning_{coord}_error = True` to surpress this warning!{bcolors.ENDC}")

def numerical_jacobian(model, x, params, eps=1e-8):
    params = np.array(params)
    J = np.zeros((len(x), len(params)))

    for j in range(len(params)):
        dp = np.zeros_like(params)
        dp[j] = eps

        f1 = np.array([model(xi, *(params + dp)) for xi in x])
        f2 = np.array([model(xi, *(params - dp)) for xi in x])

        J[:, j] = (f1 - f2) / (2 * eps)

    return J

def generic_fit(
    model: Union[Callable, Type[FitModel]],
    x_data, 
    y_data, 
    *,
    x_err=None, 
    y_err=None, 
    initial_guess= None, 
    fixed_params: Mapping[str, Any] | None = None,
    param_names =  None,
    ignore_warning_x_errors: bool = False,
    ignore_warning_y_errors: bool = False,
) -> FitResult:
    if isclass(model) and issubclass(model, FitModel):
        if not param_names:
            param_names = model.get_param_names()
        model = model.model

    x_data, y_data = filter_nan_values(x_data, y_data, warn_filter_nan=True)

    _warn_user_no_errors(x_data, x_err, ignore_warning_x_errors, "x")
    _warn_user_no_errors(y_data, y_err, ignore_warning_y_errors, "y")

    y_data, y_err = extract_vals_and_errors(y_data, y_err)
    x_data, x_err = extract_vals_and_errors(x_data, x_err)

    binding = build_fixed_param_binding(model, fixed_params=fixed_params)
    fit_model = binding.wrap_model() if binding.fixed_params else model
    fit_param_names = binding.free_param_names if binding.fixed_params else param_names
    initial_guess = order_free_initial_guess(
        model,
        initial_guess,
        binding=binding,
    )

    if binding.fixed_params and not binding.free_param_names:
        return rebuild_full_fit_result(
            binding=binding,
            free_values=[],
            free_errors=[],
            cov=np.zeros((0, 0), dtype=float),
            quality=np.nan,
            method="ODR",
        )

    # Prepare data for ODR
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)
    wrapped_model = Model(lambda B, x: _odr_wrapper(B, x, fit_model))
    odr = ODR(data, wrapped_model, beta0=initial_guess)

    # Run the fit
    out = odr.run()

    if out.res_var < 1e-12 or not np.all(np.isfinite(out.sd_beta)) or np.all(out.sd_beta < 1e-14):
    # Degenerate case
        if y_err is not None:
            print("Warning: Using error estimate fallback for fit")
            J = numerical_jacobian(fit_model, x_data, out.beta)

            W = 1 / y_err**2
            JT_W = J.T * W  # broadcasting
            cov = np.linalg.inv(JT_W @ J)

            out.sd_beta = np.sqrt(np.diag(cov))

    # return out.beta, out.sd_beta  # Best-fit parameters and uncertainties
    if binding.fixed_params:
        return rebuild_full_fit_result(
            binding=binding,
            free_values=out.beta,
            free_errors=out.sd_beta,
            cov=out.cov_beta,
            quality=out.res_var,
            method="ODR",
        )

    return generate_fit_result(fit_model, out.beta, out.sd_beta, cov=out.cov_beta, param_names=fit_param_names, quality=out.res_var, method="ODR");

def _odr_wrapper(B, x, model):
    """Wrapper to adapt curve_fit-style functions for ODR."""
    return model(x, *B)
