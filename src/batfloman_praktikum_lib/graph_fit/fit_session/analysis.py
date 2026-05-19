from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ...structs.dataset import Dataset
from ...structs.measurement import Measurement
from ..fitResult import FitResult


def _integrate_with_error(
    evaluate_nominal: Callable,
    evaluate: Callable,
    interval: tuple[float, float],
    xmin: Optional[float],
    xmax: Optional[float],
    *,
    sample_count: int,
) -> Measurement:
    resolved_xmin = interval[0] if xmin is None else float(xmin)
    resolved_xmax = interval[1] if xmax is None else float(xmax)
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


@dataclass(frozen=True)
class ComponentFitAnalysis:
    component_id: int
    order: int
    name: str
    model_name: str
    params: Dataset
    evaluate_func: Callable
    evaluate_nominal_func: Callable
    interval: tuple[float, float]
    enabled: bool = True

    def evaluate(self, x):
        return self.evaluate_func(x)

    def evaluate_nominal(self, x):
        return self.evaluate_nominal_func(x)

    def area(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        *,
        sample_count: int = 1000,
    ) -> Measurement:
        return _integrate_with_error(
            self.evaluate_nominal_func,
            self.evaluate_func,
            self.interval,
            xmin,
            xmax,
            sample_count=sample_count,
        )


@dataclass(frozen=True)
class FitAnalysis:
    fit_result: FitResult
    interval: tuple[float, float]
    model_id: int
    model_name: str
    components: list[ComponentFitAnalysis] = field(default_factory=list)

    @property
    def params(self):
        return self.fit_result.params

    @property
    def quality(self) -> float:
        return self.fit_result.quality

    @property
    def method(self) -> str:
        return self.fit_result.method

    def evaluate(self, x):
        return self.fit_result.func(x)

    def evaluate_nominal(self, x):
        return self.fit_result.func_no_err(x)

    def area(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        *,
        sample_count: int = 1000,
    ) -> Measurement:
        return _integrate_with_error(
            self.fit_result.func_no_err,
            self.fit_result.func,
            self.interval,
            xmin,
            xmax,
            sample_count=sample_count,
        )

    def component(self, ref: int | str) -> ComponentFitAnalysis:
        if isinstance(ref, int):
            for component in self.components:
                if component.component_id == ref:
                    return component
            raise KeyError(f"Unknown component id: '{ref}'.")

        matching_names = [component for component in self.components if component.name == ref]
        if len(matching_names) == 1:
            return matching_names[0]
        if len(matching_names) > 1:
            matches = ", ".join(component.name for component in matching_names)
            raise ValueError(f"Ambiguous component reference '{ref}'. Matches: {matches}")

        matching_models = [component for component in self.components if component.model_name == ref]
        if len(matching_models) == 1:
            return matching_models[0]
        if len(matching_models) > 1:
            matches = ", ".join(component.name for component in matching_models)
            raise ValueError(f"Ambiguous component reference '{ref}'. Matches: {matches}")

        raise KeyError(f"Unknown component reference: '{ref}'.")
