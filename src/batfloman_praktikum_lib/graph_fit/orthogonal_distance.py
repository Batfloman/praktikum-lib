from typing import Optional, Type, Callable, Union, Literal
from inspect import isclass
import numpy as np
import warnings, traceback
from scipy.odr import ODR, Model, RealData

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from batfloman_praktikum_lib.io.termColors import bcolors

from ..graph.plotNScatter import filter_nan_values
from .helper import extract_vals_and_errors
from .fitResult import generate_fit_result, FitResult
from .models.fitModel import FitModel
from .init_params.order_init_params import InitalParamGuess, order_initial_params

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


def generic_fit(
    model: Union[Callable, Type[FitModel]],
    x_data, 
    y_data, 
    *,
    x_err=None, 
    y_err=None, 
    initial_guess: Optional[InitalParamGuess] = None, 
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

    if initial_guess is not None:
        initial_guess = order_initial_params(model, initial_guess);
        initial_guess = [guess.value if isinstance(guess, MeasurementBase) else guess for guess in initial_guess]

    # Prepare data for ODR
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)
    wrapped_model = Model(lambda B, x: _odr_wrapper(B, x, model))
    odr = ODR(data, wrapped_model, beta0=initial_guess)

    # Run the fit
    out = odr.run()

    # return out.beta, out.sd_beta  # Best-fit parameters and uncertainties
    return generate_fit_result(model, out.beta, out.sd_beta, cov=out.cov_beta, param_names=param_names, quality=out.res_var, method="ODR");

def _odr_wrapper(B, x, model):
    """Wrapper to adapt curve_fit-style functions for ODR."""
    return model(x, *B)
