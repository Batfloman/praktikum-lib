from typing import NamedTuple
import numpy as np

from batfloman_praktikum_lib.io.formatters.formatters import custom_format

from .. import util
from .measurementBase import (
    MeasurementBase,
    ConvertibleToFloat,
    ErrorCombinationMethod,
    _combine_errors,
    _get_value_and_error,
)

class Abweichung(NamedTuple):
    sigma: float;
    percent: float;
    ratio: float;

class _MeasurementModifier:
    def __init__(self, measurement: "Measurement"):
        self._measurement = measurement

    def add_error(
        self,
        error: str | ConvertibleToFloat,
        method: ErrorCombinationMethod = "linear",
    ) -> "Measurement":
        updated = self._measurement.add_error(error, method=method)
        self._measurement.error = updated.error
        return self._measurement

    def set_error(self, error: str | ConvertibleToFloat) -> "Measurement":
        updated = self._measurement.with_error(error)
        self._measurement.error = updated.error
        return self._measurement

    def clear_error(self) -> "Measurement":
        updated = self._measurement.without_error()
        self._measurement.error = updated.error
        return self._measurement

    def round_digit(self, digits=0) -> "Measurement":
        updated = self._measurement.round_digit(digits)
        self._measurement.value = updated.value
        self._measurement.error = updated.error
        return self._measurement

    def round(self, additional_digits=0) -> "Measurement":
        updated = self._measurement.round(additional_digits)
        self._measurement.value = updated.value
        self._measurement.error = updated.error
        return self._measurement

class Measurement(MeasurementBase):
    def __init__(
        self,
        value: ConvertibleToFloat,
        uncertainty: ConvertibleToFloat | str | list[ConvertibleToFloat | str] | tuple[ConvertibleToFloat | str, ...],
        # unit=ureg.Unit(""),
        min_error=0,
        combine: ErrorCombinationMethod = "linear",
    ):
        super().__init__(value, uncertainty, min_error=min_error, combine=combine)

    @property
    def modify(self) -> _MeasurementModifier:
        return _MeasurementModifier(self)

    # ==================================================

    def add_error(
        self,
        error: str | ConvertibleToFloat,
        method: ErrorCombinationMethod = "linear",
    ) -> "Measurement":
        parsed_error = self._parse_error_value(error)
        combined_error = _combine_errors(
            [self.error, parsed_error],
            method=method,
        )
        return self._copy_with(error=combined_error)

    def with_error(self, error: str | ConvertibleToFloat) -> "Measurement":
        return self._copy_with(error=self._parse_error_value(error))

    def without_error(self) -> "Measurement":
        return self._copy_with(error=0.0)

    def clear_error(self) -> "Measurement":
        return self.without_error()

    # ==================================================

    def abweichung(self, base) -> Abweichung:
        other_val, other_err = _get_value_and_error(base)
        if other_val == 0:
            raise ValueError("Cannot calculate Abweichung to 0")

        ratio = self.value / other_val
        combined_error = (self.error**2 + other_err**2)**0.5
        if combined_error == 0:
            sigma = 0.0 if self.value == other_val else np.inf
        else:
            sigma = abs(self.value - other_val) / combined_error
        return Abweichung(
            ratio=ratio,
            percent=(ratio - 1) * 100,
            sigma=sigma,
        )

    deviation = abweichung

    # ==================================================

    def round_digit(self, digits=0) -> "Measurement":
        return self._copy_with(
            value=util.round(self.value, digits),
            error=util.ceil(self.error, digits),
        )

    def round(self, additional_digits=0) -> "Measurement":
        exponent = util.get_exponent_significant(self.error)
        return self.round_digit(-exponent + additional_digits)

    def __round__(self, decimals=0):
        return self.round_digit(decimals)

    def rint(self):
        return Measurement(np.rint(self.value), np.ceil(self.error))

    # ==================================================

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "error": self.error
        }

    @staticmethod
    def from_dict(d):
        return Measurement(d["value"], d["error"])

    # ==================================================

    def __str__(self):
        return custom_format(self)

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        return custom_format(self, format_spec=format_spec)
