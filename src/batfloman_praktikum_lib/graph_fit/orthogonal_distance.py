import numpy as np
import warnings
from scipy.odr import ODR, Model, RealData

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from ..graph.plotNScatter import filter_nan_values
from .helper import extract_vals_and_errors
from .fitResult import generate_fit_result, FitResult
from .find_initial_parameters import order_initial_params

def generic_fit(
    model, 
    x_data, 
    y_data, 
    x_err=None, 
    y_err=None, 
    initial_guess=None, 
    param_names = None
) -> FitResult:
    x_data, y_data = filter_nan_values(x_data, y_data, warn_filter_nan=True)

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
    return generate_fit_result(model, out.beta, out.sd_beta, cov=out.cov_beta, param_names=param_names, quality=out.res_var);

def _odr_wrapper(B, x, model):
    """Wrapper to adapt curve_fit-style functions for ODR."""
    return model(x, *B)
