from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...structs.measurement import Measurement
from ..fitResult import FitResult


@dataclass(frozen=True)
class FitAnalysis:
    fit_result: FitResult
    interval: tuple[float, float]
    model_id: int
    model_name: str

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
        resolved_xmin = self.interval[0] if xmin is None else float(xmin)
        resolved_xmax = self.interval[1] if xmax is None else float(xmax)
        resolved_xmin, resolved_xmax = sorted((resolved_xmin, resolved_xmax))

        if sample_count < 2:
            raise ValueError("sample_count must be at least 2.")

        x_line = np.linspace(resolved_xmin, resolved_xmax, sample_count)
        nominal = np.asarray(self.fit_result.func_no_err(x_line), dtype=float)
        lower = np.asarray(self.fit_result.min_1sigma(x_line), dtype=float)
        upper = np.asarray(self.fit_result.max_1sigma(x_line), dtype=float)

        area = float(np.trapezoid(nominal, x_line))
        lower_area = float(np.trapezoid(lower, x_line))
        upper_area = float(np.trapezoid(upper, x_line))
        error = max(upper_area - area, area - lower_area)
        return Measurement(area, error)
