from collections import namedtuple
from typing import Callable, NamedTuple, List
import numpy as np

from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.structs.dataset import Dataset

class FitResult(NamedTuple):
    func: Callable[[float], Measurement]
    params: Dataset
    quality: float
    cov: List[float]
    func_no_err: Callable[[float], float]
    min_1sigma: Callable[[float], float]
    max_1sigma: Callable[[float], float]
# FitResult = namedtuple('FitResult', ['func', 'params', 'covariance', 'value_func', 'min_func', 'max_func'])
# FitResult = namedtuple('FitResult', ['func', 'params', 'quality', 'func_no_err', 'min_1sigma', 'max_1sigma'])
# FitResult = namedtuple('FitResult', ['func', 'params', 'func_no_err', 'min_1sigma', 'max_1sigma', 'min_5sigma', 'max_5sigma'])

def generate_fit_result(model, values, errors, cov, param_names = None, quality=None) -> FitResult:
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(values))]
    elif len(param_names) < len(values):
        param_names += [f"param_{i}" for i in range(len(param_names), len(values))]

    # Create a Dataset object to hold the fit parameters and their uncertainties
    params = Dataset({
        name: Measurement(values[i], errors[i]) for i, name in enumerate(param_names)
    })

    def func_no_err(x_val):
        if isinstance(x_val, (list, np.ndarray)):
            return [model(x, *values) for x in x_val]
        return model(x_val, *values);

    def fit_func(x_val):
        if isinstance(x_val, (list, np.ndarray)):
            return [model(x, *[params[name] for name in param_names]) for x in x_val]
        return model(x_val, *[params[name] for name in param_names])

    # # Define min and max functions based on the `fit_func` uncertainty
    def min_1sigma(x_val):
        results = fit_func(x_val);
        if isinstance(results, Measurement):
            return results.value - results.error
        return [res.value - res.error for res in results]

    def max_1sigma(x_val):
        results = fit_func(x_val);
        if isinstance(results, Measurement):
            return results.value + results.error
        return [res.value + res.error for res in results]

    return FitResult(
        func=fit_func,
        params=params,
        quality=quality,
        cov=cov,
        func_no_err=func_no_err,
        min_1sigma=min_1sigma,
        max_1sigma=max_1sigma
    )
