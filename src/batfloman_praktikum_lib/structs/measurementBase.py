import numpy as np
from typing import Union, Tuple, Sequence

# ==================================================

ConvertibleToFloat = Union[float, int, np.integer, np.floating]

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
    elif isinstance(other, ConvertibleToFloat):
        return (float(other), 0.0)
    raise TypeError(f"Unsupported type: {type(other)}")

def _get_value(other):
    if isinstance(other, MeasurementBase):
        return other.value
    elif isinstance(other, ConvertibleToFloat):
        return float(other)
    raise TypeError(f"Unsupported type: {type(other)}")

# ==================================================
#    Class
# ==================================================

class MeasurementBase:
    def __init__(self, value: float, uncertainties: str | ConvertibleToFloat | Sequence[str | ConvertibleToFloat], min_error=0):
        self.value = float(value)

        if not isinstance(uncertainties, (list, tuple)):
            uncertainties = [uncertainties]

        errors = []
        for u in uncertainties:
            if isinstance(u, str):
                u = _parse_uncertainty_str(value, u)
            errors.append(float(u))

# Fehler kombinieren
        total_error = sum(e for e in errors)
        self.error = max(total_error, min_error)

    # def __init__(self, value: ConvertibleToFloat, uncertainty: ConvertibleToFloat | str, min_error=0):
    #     self.value = float(value)
    #
    #     if isinstance(uncertainty, str):
    #         uncertainty = _parse_uncertainty_str(value, uncertainty);
    #     uncertainty = max(float(uncertainty), min_error)
    #     self.error = uncertainty

    @classmethod
    def from_value_error(cls, value, error):
        return cls(value, error)

    # ==================================================
    #     Comparison operations
    # ==================================================

    def __lt__(self, other):
        other_val = _get_value(other)
        return NotImplemented if other_val is NotImplemented else (self.value < other_val)

    def __le__(self, other):
        other_val = _get_value(other)
        return NotImplemented if other_val is NotImplemented else (self.value <= other_val)

    def __eq__(self, other):
        other_val = _get_value(other)
        return NotImplemented if other_val is NotImplemented else (self.value == other_val)

    def __ge__(self, other):
        other_val = _get_value(other)
        return NotImplemented if other_val is NotImplemented else (self.value >= other_val)

    def __gt__(self, other):
        other_val = _get_value(other)
        return NotImplemented if other_val is NotImplemented else (self.value > other_val)

    # ==================================================
    #     Math Operations
    # ==================================================

    # ==================================================
    # addition

    def __add__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value + other_val
        new_uncertainty = np.hypot(self.error, other_err)

        return self.__class__.from_value_error(new_value, new_uncertainty)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value - other_val
        new_uncertainty = np.hypot(self.error, other_err)

        return self.__class__.from_value_error(new_value, new_uncertainty)

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
            
        return self.__class__.from_value_error(new_value, new_uncertainty)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = self.value / other_val
        t1 = (self.error / other_val)**2
        t2 = (self.value * other_err / (other_val**2))**2
        new_uncertainty = np.sqrt(t1 + t2)

        return self.__class__.from_value_error(new_value, new_uncertainty)

    def __rtruediv__(self, other):
        other_val, other_err = _get_value_and_error(other)

        new_value = other_val / self.value
        t1 = (other_err / self.value)**2
        t2 = (other_val * self.error / (self.value)**2)**2
        new_uncertainty = np.sqrt(t1 + t2)

        return self.__class__.from_value_error(new_value, new_uncertainty)

    # ==================================================
    # operators & advanced func

    def __neg__(self):
        return self.__class__.from_value_error(-self.value, self.error)

    def __abs__(self):
        return self.__class__.from_value_error(abs(self.value), self.error)

    def __mod__(self, other):
        val = self.value % other
        err = self.error
        return self.__class__.from_value_error(val, err)

    def __pow__(self, other):
        other_val, other_err = _get_value_and_error(other)

        value = self.value ** other_val
        t1 = other_val * self.value ** (other_val - 1) * self.error
        t2 = value * np.log(self.value) * other_err
        error = np.sqrt(t1**2 + t2**2)

        return self.__class__.from_value_error(value, error)
    
    def __rpow__(self, other):
        other_val, other_err = _get_value_and_error(other)

        value = other_val ** self.value
        t1 = self.value * other_val**(self.value - 1) * other_err
        t2 = value * np.log(other_val) * self.error
        error = np.sqrt(t1**2 + t2**2)

        return self.__class__.from_value_error(value, error)

    # ==================================================
    # numpy compatibility

    def rint(self):
        return self.__class__.from_value_error(np.rint(self.value), np.ceil(self.error))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy universal functions.
        """
        if method != "__call__":
            return NotImplemented
        
        # Extract the input values and errors
        values = [_get_value(i) for i in inputs]
        errors = [i.error if isinstance(i, MeasurementBase) else 0 for i in inputs]

        # Handle specific ufuncs
        match ufunc:
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
                val = np.sqrt(values[0]);
                err = .5 * errors[0] / np.sqrt(values[0])
            case np.sin:
                val = np.sin(values[0])
                err = np.abs(np.cos(values[0]) * errors[0])
            case np.cos:
                val = np.cos(values[0])
                err = np.abs(np.sin(values[0]) * errors[0])
            case np.arcsin:
                val = np.arcsin(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
            case np.arccos:
                val = np.arccos(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
            case np.tan:
                val = np.tan(values[0])
                err = np.abs((1 / np.cos(values[0])**2) * errors[0])
            case np.exp:
                val = np.exp(values[0]);
                err = np.exp(values[0]) * errors[0];
            case np.log:
                # Propagate the value and error for log(x)
                val = np.log(values[0])
                err = errors[0] / values[0]  # Error propagation formula for log
            case np.log10:
                # Propagate the value and error for log(x)
                val = np.log10(values[0])
                err = errors[0] / values[0] / np.log(10) # Error propagation formula for n-log
            case np.rad2deg:
                val = np.rad2deg(values[0]);
                err = np.rad2deg(errors[0]);
            case np.deg2rad:
                val = np.deg2rad(values[0]);
                err = np.deg2rad(errors[0]);
            case np.abs:
                val = np.abs(values[0])
                err = errors[0]
            case _:
                raise NotImplementedError(f"not handled function: {ufunc}")
        return self.__class__.from_value_error(val, err)

    # ==================================================

    def __str__(self):
        return f"{self.value} \pm {self.error}";

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        try:
            return f"{self.value:{format_spec}} ± {self.error:{format_spec}}"
        except ValueError:
            return str(self)

