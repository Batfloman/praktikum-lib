from collections import namedtuple
from typing import Callable, NamedTuple, List, Literal
import numpy as np

from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.structs.dataset import Dataset

type FIT_METHODS = Literal["least_squares", "ODR", "idk"]

def _get_quality_statement(quality):
    if quality > 2:
        return  "(Residuen zu groß / Modell passt schlecht)"
    elif quality > 1.2:
        return "(Residuen leicht größer als erwartet)"
    elif quality < 0.5:
        return "(Residuen zu klein / Unsicherheiten überschätzt)"
    elif quality < 0.8:
        return "(Residuen etwas kleiner als erwartet)"
    else:
        return ""  # alles ok

class FitResult(NamedTuple):
    func: Callable[[float], Measurement]
    params: Dataset
    quality: float
    cov: List[float]
    func_no_err: Callable[[float], float]
    min_1sigma: Callable[[float], float]
    max_1sigma: Callable[[float], float]
    method: FIT_METHODS

    def __repr__(self):
        return (
            "FitResult(\n"
            f"  method  = {self.method}\n"
            f"  quality = {self.quality:.3f} {_get_quality_statement(self.quality)}\n"
            f"  params  = {{{self.params}}}\n"
            f"  cov=\n{self.cov}\n"
            f"  func        = {self.func}\n"
            f"  func_no_err = {self.func_no_err}\n"
            f"  min_1sigma  = {self.min_1sigma}\n"
            f"  max_1sigma  = {self.max_1sigma}\n"
            ")"
        )

def generate_fit_result(model, values, errors, cov, 
    param_names = None, 
    quality=None, 
    method: FIT_METHODS = "idk",
) -> FitResult:
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
        max_1sigma=max_1sigma,
        method = method,
    )
