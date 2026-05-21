from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import isclass
import inspect
from typing import Any, Callable, Type

import numpy as np

from .fitResult import FitResult, generate_fit_result
from .init_params._helper import get_model_fn
from .init_params.order_init_params import InitalParamGuess, order_initial_params
from .models.fitModel import FitModel
from ..structs.measurementBase import MeasurementBase


@dataclass(frozen=True)
class FixedParamBinding:
    model: Callable
    full_param_names: list[str]
    free_param_names: list[str]
    fixed_params: dict[str, float]

    def merge_free_values(self, free_values) -> list[float]:
        free_lookup = {
            param_name: free_values[idx]
            for idx, param_name in enumerate(self.free_param_names)
        }
        return [
            self.fixed_params[param_name]
            if param_name in self.fixed_params
            else free_lookup[param_name]
            for param_name in self.full_param_names
        ]

    def wrap_model(self) -> Callable:
        full_param_names = list(self.full_param_names)
        fixed_params = dict(self.fixed_params)
        free_param_names = list(self.free_param_names)
        model = self.model

        def wrapped(x, *free_values):
            free_lookup = {
                param_name: free_values[idx]
                for idx, param_name in enumerate(free_param_names)
            }
            ordered_values = [
                fixed_params[param_name]
                if param_name in fixed_params
                else free_lookup[param_name]
                for param_name in full_param_names
            ]
            return model(x, *ordered_values)

        return wrapped


def get_param_names(model: Callable | Type[FitModel]) -> list[str]:
    if isclass(model) and issubclass(model, FitModel):
        return list(model.get_param_names())
    model_fn = get_model_fn(model)
    return list(inspect.signature(model_fn).parameters.keys())[1:]


def normalize_fixed_params(
    model: Callable | Type[FitModel],
    fixed_params: Mapping[str, Any] | None,
) -> dict[str, float]:
    if fixed_params is None:
        return {}

    param_names = get_param_names(model)
    unknown_params = [name for name in fixed_params if name not in param_names]
    if unknown_params:
        raise ValueError(
            f"Unknown fixed parameter(s): {unknown_params}. Expected parameters: {param_names}"
        )

    normalized: dict[str, float] = {}
    for name, value in fixed_params.items():
        if isinstance(value, MeasurementBase):
            normalized[name] = float(value.value)
        else:
            normalized[name] = float(value)
    return normalized


def build_fixed_param_binding(
    model: Callable | Type[FitModel],
    *,
    fixed_params: Mapping[str, Any] | None,
) -> FixedParamBinding:
    model_fn = get_model_fn(model)
    full_param_names = get_param_names(model)
    normalized_fixed_params = normalize_fixed_params(model, fixed_params)
    free_param_names = [
        param_name
        for param_name in full_param_names
        if param_name not in normalized_fixed_params
    ]
    return FixedParamBinding(
        model=model_fn,
        full_param_names=full_param_names,
        free_param_names=free_param_names,
        fixed_params=normalized_fixed_params,
    )


def order_free_initial_guess(
    model: Callable | Type[FitModel],
    initial_guess: InitalParamGuess | None,
    *,
    binding: FixedParamBinding,
) -> list[float] | None:
    if initial_guess is None:
        return None

    if not binding.fixed_params:
        ordered_guess = order_initial_params(model, initial_guess)
        return [
            guess.value if isinstance(guess, MeasurementBase) else guess
            for guess in ordered_guess
        ]

    if isinstance(initial_guess, Mapping):
        return [
            initial_guess[param_name].value if isinstance(initial_guess[param_name], MeasurementBase) else initial_guess[param_name]
            for param_name in binding.free_param_names
        ]

    if isinstance(initial_guess, Sequence) and not isinstance(initial_guess, (str, bytes, Mapping)):
        ordered_guess = list(initial_guess)
        if len(ordered_guess) == len(binding.free_param_names):
            return [
                guess.value if isinstance(guess, MeasurementBase) else guess
                for guess in ordered_guess
            ]
        if len(ordered_guess) == len(binding.full_param_names):
            free_indices = [
                idx
                for idx, param_name in enumerate(binding.full_param_names)
                if param_name not in binding.fixed_params
            ]
            return [
                ordered_guess[idx].value if isinstance(ordered_guess[idx], MeasurementBase) else ordered_guess[idx]
                for idx in free_indices
            ]

    ordered_guess = order_initial_params(model, initial_guess)
    free_indices = [
        idx
        for idx, param_name in enumerate(binding.full_param_names)
        if param_name not in binding.fixed_params
    ]
    return [
        ordered_guess[idx].value if isinstance(ordered_guess[idx], MeasurementBase) else ordered_guess[idx]
        for idx in free_indices
    ]


def rebuild_full_fit_result(
    *,
    binding: FixedParamBinding,
    free_values,
    free_errors,
    cov,
    quality,
    method,
) -> FitResult:
    full_values = binding.merge_free_values(free_values)
    free_error_lookup = {
        param_name: float(free_errors[idx])
        for idx, param_name in enumerate(binding.free_param_names)
    }
    full_errors = np.asarray(
        [
            0.0 if param_name in binding.fixed_params else free_error_lookup[param_name]
            for param_name in binding.full_param_names
        ],
        dtype=float,
    )

    if cov is None:
        full_cov = None
    else:
        full_cov = np.zeros((len(binding.full_param_names), len(binding.full_param_names)), dtype=float)
        free_index_lookup = {
            param_name: idx
            for idx, param_name in enumerate(binding.free_param_names)
        }
        for row_idx, row_name in enumerate(binding.full_param_names):
            if row_name in binding.fixed_params:
                continue
            for col_idx, col_name in enumerate(binding.full_param_names):
                if col_name in binding.fixed_params:
                    continue
                full_cov[row_idx, col_idx] = cov[
                    free_index_lookup[row_name],
                    free_index_lookup[col_name],
                ]

    return generate_fit_result(
        binding.model,
        full_values,
        full_errors,
        cov=full_cov,
        param_names=list(binding.full_param_names),
        quality=quality,
        method=method,
    )
