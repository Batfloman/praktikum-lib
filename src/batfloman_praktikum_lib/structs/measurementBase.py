import numpy as np
from scipy.special import erf
import re

from typing import Literal, Sequence, Tuple, TypeAlias, Union

# ==================================================

ConvertibleToFloat = Union[float, int, np.integer, np.floating]
ErrorCombinationMethod: TypeAlias = Literal["linear", "quadrature"]

_PARENTHESIZED_MEASUREMENT_PATTERN = re.compile(
    r"^\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\("
    r"(?P<error>\d+)"
    r"\)"
    r"\s*$"
)


def _parse_measurement_literal(literal: str) -> tuple[float, float] | None:
    match = _PARENTHESIZED_MEASUREMENT_PATTERN.match(literal)
    if match is None:
        return None

    value_str = match.group("value")
    error_digits = match.group("error")

    value = float(value_str)

    mantissa = value_str
    exponent = 0
    if "e" in mantissa.lower():
        mantissa, exponent_str = re.split(r"[eE]", mantissa, maxsplit=1)
        exponent = int(exponent_str)

    if "." in mantissa:
        decimal_places = len(mantissa.split(".", maxsplit=1)[1])
    else:
        decimal_places = 0

    error = float(error_digits) * 10 ** (exponent - decimal_places)
    return value, error

def _parse_uncertainty_str(value: float, uncertainty: str) -> float:
    uncertainty = uncertainty.replace("\"", "")
    if uncertainty.strip().endswith("%"):
        try:
            percentage = float(uncertainty[:-1])
            return abs(value) * percentage / 100
        except ValueError:
            return np.nan
    else:
        try:
            return float(uncertainty)
        except ValueError:
            return np.nan

def _get_value_and_error(other) -> Tuple[float, float]:
    if isinstance(other, MeasurementBase):
        return (other.value, other.error)
    elif isinstance(other, np.ndarray):
        if other.dtype != object:
            values = other.astype(float, copy=False)
            errors = np.zeros_like(values, dtype=float)
            return (values, errors)

        values = np.vectorize(lambda item: _get_value_and_error(item)[0], otypes=[float])(other)
        errors = np.vectorize(lambda item: _get_value_and_error(item)[1], otypes=[float])(other)
        return (values, errors)
    elif isinstance(other, ConvertibleToFloat):
        return (float(other), 0.0)
    raise TypeError(f"Unsupported type: {type(other)}")

def _combine_errors(
    errors: Sequence[float],
    method: ErrorCombinationMethod = "linear",
) -> float:
    match method:
        case "linear":
            return sum(abs(error) for error in errors)
        case "quadrature":
            return float(np.sqrt(sum(error**2 for error in errors)))
        case _:
            raise ValueError(f"Unknown error combination method: {method}")

def _get_value(other):
    if isinstance(other, MeasurementBase):
        return other.value
    elif isinstance(other, np.ndarray):
        return _get_value_and_error(other)[0]
    elif isinstance(other, ConvertibleToFloat):
        return float(other)
    raise TypeError(f"Unsupported type: {type(other)}")


def _safe_power(base, exp):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.power(base, exp)

# ==================================================
#    Class
# ==================================================

class MeasurementBase:
    __array_priority__ = 10000

    def __init__(
        self,
        value: float | str,
        uncertainties: str | ConvertibleToFloat | Sequence[str | ConvertibleToFloat] | None = None,
        min_error=0,
        combine: ErrorCombinationMethod = "linear",
    ):
        if uncertainties is None:
            if not isinstance(value, str):
                raise TypeError("uncertainties must be provided unless value uses '1.23(4)' syntax")

            parsed_measurement = _parse_measurement_literal(value)
            if parsed_measurement is None:
                raise TypeError("uncertainties must be provided unless value uses '1.23(4)' syntax")

            value, uncertainties = parsed_measurement

        self.value = float(value)

        if not isinstance(uncertainties, (list, tuple)):
            uncertainties = [uncertainties]

        errors = []
        for u in uncertainties:
            if isinstance(u, str):
                u = _parse_uncertainty_str(value, u)
            errors.append(float(u))

        # Combine individual uncertainty contributions according to the
        # configured convention for this measurement.
        total_error = _combine_errors(errors, method=combine)
        self.error = max(total_error, min_error)

    @classmethod
    def from_value_error(cls, value, error):
        return cls(value, error)

    def _from_value_error(self, value: float, error: float):
        return self.__class__.from_value_error(value, error)

    def _parse_error_value(self, error: str | ConvertibleToFloat) -> float:
        if isinstance(error, str):
            error = _parse_uncertainty_str(self.value, error)
        return float(error)

    def _copy_with(self, *, value: float | None = None, error: float | None = None):
        return self.__class__(
            self.value if value is None else value,
            self.error if error is None else error,
        )

    # ==================================================
    #     Comparison operations
    # ==================================================

    def __lt__(self, other):
        return self.value < _get_value(other)

    def __le__(self, other):
        return self.value <= _get_value(other)

    def __eq__(self, other):
        return self.value == _get_value(other)

    def __ge__(self, other):
        return self.value >= _get_value(other)

    def __gt__(self, other):
        return self.value > _get_value(other)

    # ==================================================
    #     Math Operations
    # ==================================================

    # ==================================================
    # addition

    def __add__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value + other_val
        new_uncertainty = np.hypot(self.error, other_err)

        return self._from_value_error(new_value, new_uncertainty)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value - other_val
        new_uncertainty = np.hypot(self.error, other_err)

        return self._from_value_error(new_value, new_uncertainty)

    def __rsub__(self, other):
        return (-self).__add__(other)

    # ==================================================
    # multiplication

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([self * o for o in other])

        other_val, other_err = _get_value_and_error(other)

        new_value = self.value * other_val
        t1 = (self.value * other_err)**2
        t2 = (self.error * other_val)**2
        new_uncertainty = np.sqrt(t1 + t2)

        return self._from_value_error(new_value, new_uncertainty)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value / other_val
        t1 = (self.error / other_val)**2
        t2 = (self.value * other_err / (other_val**2))**2
        new_uncertainty = np.sqrt(t1 + t2)

        return self._from_value_error(new_value, new_uncertainty)

    def __rtruediv__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = other_val / self.value
        t1 = (other_err / self.value)**2
        t2 = (other_val * self.error / (self.value)**2)**2
        new_uncertainty = np.sqrt(t1 + t2)

        return self._from_value_error(new_value, new_uncertainty)

    # ==================================================
    # operators & advanced func

    def __neg__(self):
        return self._from_value_error(-self.value, self.error)

    def __abs__(self):
        return self._from_value_error(abs(self.value), self.error)

    def __mod__(self, other):
        val = self.value % other
        err = self.error
        return self._from_value_error(val, err)

    def __pow__(self, other):
        other_val, other_err = _get_value_and_error(other)

        value = _safe_power(self.value, other_val)
        t1 = other_val * _safe_power(self.value, other_val - 1) * self.error

        t2 = 0.0
        if other_err != 0:
            if self.value > 0:
                t2 = other_err * value * np.log(self.value)
            else:
                t2 = np.nan

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            error = np.sqrt(t1**2 + t2**2)

        return self._from_value_error(value, error)

    def __rpow__(self, other):
        other_val, other_err = _get_value_and_error(other)

        value = _safe_power(other_val, self.value)
        t1 = self.value * _safe_power(other_val, self.value - 1) * other_err

        t2 = np.nan
        if self.error == 0:
            t2 = 0.0
        elif other_val > 0:
            t2 = value * np.log(other_val) * self.error

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            error = np.sqrt(t1**2 + t2**2)

        return self._from_value_error(value, error)

    # ==================================================
    # numpy compatibility

    def sin(self):
        val = np.sin(self.value)
        err = np.abs(np.cos(self.value) * self.error)

        return self._from_value_error(val, err)

    def sqrt(self):
        val = np.sqrt(self.value)
        err = 0.5 * self.error / val
        return self._from_value_error(val, err)

    def log(self):
        return np.log(self)

    def log10(self):
        return np.log10(self)

    def deg2rad(self):
        val = np.deg2rad(self.value)
        err = np.deg2rad(self.error)
        return self._from_value_error(val, err)

    def rint(self):
        return self._from_value_error(np.rint(self.value), np.ceil(self.error))

    def __float__(self):
        return float(self.value)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        values = []
        errors = []
        for item in inputs:
            value, error = _get_value_and_error(item)
            values.append(value)
            errors.append(error)

        match ufunc:
            # --------------------
            # math operations

            case np.add:
                val = values[0] + values[1]
                d1 = errors[0]
                d2 = errors[1]
                err = (d1**2 + d2**2)**.5
            case np.subtract:
                val = values[0] - values[1]
                d1 = errors[0]
                d2 = -errors[1]
                err = (d1**2 + d2**2)**.5

            case np.multiply:
                val = values[0] * values[1]
                d1 = (errors[0] * values[1])
                d2 = (values[0] * errors[1])
                err = (d1**2 + d2**2)**.5
            case np.divide:
                val = values[0] / values[1]
                d1 = errors[0] / values[1]
                d2 = - values[0] * errors[1] / values[1]**2
                err = (d1**2 + d2**2)**.5

            case np.sqrt:
                val = np.sqrt(values[0])
                err = .5 * errors[0] / np.sqrt(values[0])
            case np.power:
                base = values[0]
                exp = values[1]
                val = _safe_power(base, exp)
                d_base = exp * _safe_power(base, exp - 1) * errors[0]

                d_exp = 0.0
                if np.any(np.asarray(errors[1]) != 0):
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        d_exp = np.where(
                            base > 0,
                            val * np.log(base) * errors[1],
                            np.nan,
                        )

                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    err = np.sqrt(d_base**2 + d_exp**2)

            # --------------------
            # trig

            case np.sin:
                val = np.sin(values[0])
                err = np.abs(np.cos(values[0]) * errors[0])
            case np.cos:
                val = np.cos(values[0])
                err = np.abs(np.sin(values[0]) * errors[0])
            case np.tan:
                val = np.tan(values[0])
                err = errors[0] / np.cos(values[0])**2

            case np.arcsin:
                val = np.arcsin(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
            case np.arccos:
                val = np.arccos(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
            case np.arctan:
                val = np.arctan(values[0])
                err = errors[0] / (1 + values[0]**2)

            case np.sinh:
                val = np.sinh(values[0])
                err = np.abs(np.cosh(values[0]) * errors[0])
            case np.cosh:
                val = np.cosh(values[0])
                err = np.abs(np.sinh(values[0]) * errors[0])
            case np.tanh:
                val = np.tanh(values[0])
                err = errors[0] / np.cosh(values[0])**2

            case np.arcsinh:
                val = np.arcsinh(values[0])
                err = errors[0] / np.sqrt(1 + values[0]**2)
            case np.arccosh:
                val = np.arccosh(values[0])
                err = errors[0] / np.sqrt(values[0] - 1) / np.sqrt(values[0] + 1)
            case np.arctanh:
                val = np.arctanh(values[0])
                err = errors[0] / (1 - values[0]**2)

            # --------------------
            # exp,log

            case np.exp:
                val = np.exp(values[0])
                err = np.exp(values[0]) * errors[0]
            case np.log:
                val = np.log(values[0])
                err = errors[0] / values[0]
            case np.log10:
                val = np.log10(values[0])
                err = errors[0] / values[0] / np.log(10)
            case np.logaddexp:
                val = np.logaddexp(values[0], values[1])
                d1 = np.exp(values[0] - val) * errors[0]
                d2 = np.exp(values[1] - val) * errors[1]
                err = (d1**2 + d2**2)**.5

            # --------------------
            # speical boys 

            case _ if ufunc is erf:
                val = erf(values[0])
                err = np.abs((2 / np.sqrt(np.pi)) * np.exp(-values[0]**2) * errors[0])

            # --------------------
            # modifications

            case np.rad2deg:
                val = np.rad2deg(values[0])
                err = np.rad2deg(errors[0])
            case np.deg2rad:
                val = np.deg2rad(values[0])
                err = np.deg2rad(errors[0])
            case np.abs:
                val = np.abs(values[0])
                err = errors[0]
            case np.minimum:
                val = np.minimum(values[0], values[1])
                err = np.where(
                    (values[0] <= values[1]) | np.isnan(values[0]),
                    errors[0],
                    errors[1],
                )
            case np.maximum:
                val = np.maximum(values[0], values[1])
                err = np.where(
                    (values[0] >= values[1]) | np.isnan(values[0]),
                    errors[0],
                    errors[1],
                )
            case np.isnan:
                return np.isnan(values[0])
            case np.less:
                return values[0] < values[1]
            case np.less_equal:
                return values[0] <= values[1]
            case np.equal:
                return values[0] == values[1]
            case np.not_equal:
                return values[0] != values[1]
            case np.greater_equal:
                return values[0] >= values[1]
            case np.greater:
                return values[0] > values[1]
            case _:
                raise NotImplementedError(f"not handled function: {ufunc}")

        if np.ndim(val) == 0 and np.ndim(err) == 0:
            return self._from_value_error(float(val), float(err))

        val, err = np.broadcast_arrays(val, err)
        return np.vectorize(self._from_value_error, otypes=[object])(val, err)

    # ==================================================

    def __str__(self):
        return fr"{self.value} \pm {self.error}"

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        try:
            return f"{self.value:{format_spec}} ± {self.error:{format_spec}}"
        except ValueError:
            return str(self)
