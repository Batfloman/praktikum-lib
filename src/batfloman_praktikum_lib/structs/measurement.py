from typing import NamedTuple
import numpy as np

from .. import util
from .measurementBase import (
    MeasurementBase,
    ConvertibleToFloat,
    ErrorCombinationMethod,
    _combine_errors,
    _get_value_and_error,
)

class Deviation(NamedTuple):
    sigma: float
    percent: float
    ratio: float

class _MeasurementModifier:
    def __init__(self, measurement: "Measurement"):
        self._measurement = measurement

    def add_error(
        self,
        error: str | ConvertibleToFloat,
        method: ErrorCombinationMethod = "linear",
    ) -> "Measurement":
        """In place: Add an uncertainty to the current measurement."""
        updated = self._measurement.add_error(error, method=method)
        self._measurement.error = updated.error
        return self._measurement

    def set_error(self, error: str | ConvertibleToFloat) -> "Measurement":
        """In place: Replace the current uncertainty."""
        updated = self._measurement.with_error(error)
        self._measurement.error = updated.error
        return self._measurement

    def clear_error(self) -> "Measurement":
        """In place: Remove the current uncertainty."""
        updated = self._measurement.without_error()
        self._measurement.error = updated.error
        return self._measurement

    def round_digit(self, digits=0) -> "Measurement":
        """In place: Round value and uncertainty to a fixed decimal position."""
        updated = self._measurement.round_digit(digits)
        self._measurement.value = updated.value
        self._measurement.error = updated.error
        return self._measurement

    def round(self, additional_digits=0) -> "Measurement":
        """In place: Round the measurement based on the uncertainty magnitude."""
        updated = self._measurement.round(additional_digits)
        self._measurement.value = updated.value
        self._measurement.error = updated.error
        return self._measurement

class Measurement(MeasurementBase):
    def __init__(
        self,
        value: ConvertibleToFloat,
        uncertainty: ConvertibleToFloat | str | list[ConvertibleToFloat | str] | tuple[ConvertibleToFloat | str, ...],
        min_error=0,
        combine: ErrorCombinationMethod = "linear",
    ):
        """Create a measurement from a value and one or more uncertainty terms."""
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
        """Return a new measurement with an additional uncertainty contribution."""
        parsed_error = self._parse_error_value(error)
        combined_error = _combine_errors(
            [self.error, parsed_error],
            method=method,
        )
        return self._copy_with(error=combined_error)

    def with_error(self, error: str | ConvertibleToFloat) -> "Measurement":
        """Return a new measurement with its uncertainty replaced."""
        return self._copy_with(error=self._parse_error_value(error))

    def without_error(self) -> "Measurement":
        """Return a new measurement without uncertainty."""
        return self._copy_with(error=0.0)

    # ==================================================

    def deviation(self, base) -> Deviation:
        """Compare this measurement to a reference value or measurement."""
        other_val, other_err = _get_value_and_error(base)
        if other_val == 0:
            raise ValueError("Cannot calculate Abweichung to 0")

        ratio = self.value / other_val

        combined_error = (self.error**2 + other_err**2)**0.5
        if combined_error == 0:
            sigma = 0.0 if self.value == other_val else np.inf
        else:
            sigma = abs(self.value - other_val) / combined_error

        return Deviation(
            ratio=ratio,
            percent=(ratio - 1) * 100,
            sigma=sigma,
        )

    # ==================================================

    def round(self, additional_digits=0) -> "Measurement":
        """Return a new measurement rounded relative to its uncertainty."""
        exponent = util.get_exponent_significant(self.error)
        return self.round_digit(-exponent + additional_digits)

    def round_digit(self, digits=0) -> "Measurement":
        """Return a new measurement rounded to a fixed decimal position."""
        return self._copy_with(
            value=util.round(self.value, digits),
            error=util.ceil(self.error, digits),
        )

    def __round__(self, decimals=0):
        return self.round_digit(decimals)

    def rint(self):
        """Return a new measurement rounded to integer values."""
        return self._copy_with(
            value=np.rint(self.value),
            error=np.ceil(self.error),
        )

    # ==================================================

    def __str__(self):
        from batfloman_praktikum_lib.io.formatters.formatters import custom_format
        return custom_format(self)

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        from batfloman_praktikum_lib.io.formatters.formatters import custom_format
        return custom_format(self, format_spec=format_spec)

    def to_dict(self) -> dict:
        return {
            "__type__": "Measurement",
            "value": self.value,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Measurement":
        return cls(data["value"], data["error"])

    def to_json(self, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import dumps_json
        return dumps_json(self, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Measurement":
        from batfloman_praktikum_lib.io.json import loads_json
        return loads_json(json_str)

    def save_json(self, path: str, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import save_json
        return save_json(self, path, indent=indent)

    @classmethod
    def load_json(cls, path: str) -> "Measurement":
        from batfloman_praktikum_lib.io.json import load_json
        return load_json(path)
