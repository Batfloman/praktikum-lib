from typing import Callable, Literal

import numpy as np
from scipy.special import erf

from ...structs.dataset import Dataset
from ...structs.measurement import Measurement

AreaBounds = Literal["interval", "full", "positive", "negative"] | tuple[float, float]


def integrate_with_error(
    evaluate_nominal: Callable,
    evaluate: Callable,
    bounds: tuple[float, float],
    *,
    sample_count: int,
) -> Measurement:
    resolved_xmin = float(bounds[0])
    resolved_xmax = float(bounds[1])
    resolved_xmin, resolved_xmax = sorted((resolved_xmin, resolved_xmax))

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    x_line = np.linspace(resolved_xmin, resolved_xmax, sample_count)
    nominal = np.asarray(evaluate_nominal(x_line), dtype=float)
    measured = evaluate(x_line)
    lower = np.asarray(
        [value.value - value.error for value in measured],
        dtype=float,
    )
    upper = np.asarray(
        [value.value + value.error for value in measured],
        dtype=float,
    )

    area = float(np.trapezoid(nominal, x_line))
    lower_area = float(np.trapezoid(lower, x_line))
    upper_area = float(np.trapezoid(upper, x_line))
    error = max(upper_area - area, area - lower_area)
    return Measurement(area, error)


def resolve_finite_bounds(
    interval: tuple[float, float],
    bounds: AreaBounds,
) -> tuple[float, float]:
    if bounds == "interval":
        return interval
    if isinstance(bounds, tuple):
        return (float(bounds[0]), float(bounds[1]))
    raise ValueError(f"Bounds '{bounds}' do not define a finite integration interval.")


def gaussian_area_full(params: Dataset) -> Measurement:
    amplitude = _gaussian_param(params, "A")
    sigma = _gaussian_param(params, "sigma")
    return amplitude * sigma * np.sqrt(2 * np.pi)


def gaussian_area_positive(params: Dataset) -> Measurement:
    amplitude = _gaussian_param(params, "A")
    sigma = _gaussian_param(params, "sigma")
    x0 = _gaussian_param(params, "x0")
    return amplitude * sigma * np.sqrt(np.pi / 2) * (1 + erf(x0 / (np.sqrt(2) * sigma)))


def gaussian_area_negative(params: Dataset) -> Measurement:
    amplitude = _gaussian_param(params, "A")
    sigma = _gaussian_param(params, "sigma")
    x0 = _gaussian_param(params, "x0")
    return amplitude * sigma * np.sqrt(np.pi / 2) * (1 - erf(x0 / (np.sqrt(2) * sigma)))


def component_area_special_bounds(
    *,
    model_name: str,
    params: Dataset,
    bounds: Literal["full", "positive", "negative"],
) -> Measurement:
    if model_name != "Gaussian":
        raise NotImplementedError(
            f"Bounds '{bounds}' are not implemented for component type '{model_name}'."
        )
    if bounds == "full":
        return gaussian_area_full(params)
    if bounds == "positive":
        return gaussian_area_positive(params)
    return gaussian_area_negative(params)


def _gaussian_param(params: Dataset, name: str):
    if name not in params:
        raise ValueError(f"Gaussian component is missing parameter '{name}'.")
    return params[name]
