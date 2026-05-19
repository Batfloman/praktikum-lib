from dataclasses import dataclass, field
from typing import Callable

from ...structs.dataset import Dataset
from ...structs.measurement import Measurement
from ..fitResult import FitResult
from .area import (
    AreaBounds,
    component_area_special_bounds,
    integrate_with_error,
    resolve_finite_bounds,
)


@dataclass(frozen=True)
class ComponentFitAnalysis:
    component_id: int
    order: int
    name: str
    model_name: str
    model_type: object | None
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
        bounds: AreaBounds,
        *,
        sample_count: int = 1000,
    ) -> Measurement:
        if bounds in {"full", "positive", "negative"}:
            return component_area_special_bounds(
                model_name=self.model_name,
                params=self.params,
                bounds=bounds,
            )

        return integrate_with_error(
            self.evaluate_nominal_func,
            self.evaluate_func,
            resolve_finite_bounds(self.interval, bounds),
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
        bounds: AreaBounds,
        *,
        sample_count: int = 1000,
    ) -> Measurement:
        if bounds in {"full", "positive", "negative"}:
            if not self.components:
                raise NotImplementedError(
                    f"Bounds '{bounds}' are not available for this fit analysis."
                )
            total = Measurement(0, 0)
            for component in self.components:
                total = total + component.area(bounds, sample_count=sample_count)
            return total

        return integrate_with_error(
            self.fit_result.func_no_err,
            self.fit_result.func,
            resolve_finite_bounds(self.interval, bounds),
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
