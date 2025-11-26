import numpy as np
from scipy.odr import ODR, Model, RealData

from .fitResult import generate_fit_result, FitResult

def generic_fit(model, x_data, y_data, xerr=None, yerr=None, initial_guess=None, param_names = None) -> FitResult:
    if np.all(yerr == 0):
        yerr = None

    # Prepare data for ODR
    data = RealData(x_data, y_data, sx=xerr, sy=yerr)
    wrapped_model = Model(lambda B, x: _odr_wrapper(B, x, model))
    odr = ODR(data, wrapped_model, beta0=initial_guess)

    # Run the fit
    out = odr.run()

    # return out.beta, out.sd_beta  # Best-fit parameters and uncertainties
    return generate_fit_result(model, out.beta, out.sd_beta, cov=out.cov_beta, param_names=param_names, quality=out.res_var);

def _odr_wrapper(B, x, model):
    """Wrapper to adapt curve_fit-style functions for ODR."""
    return model(x, *B)
