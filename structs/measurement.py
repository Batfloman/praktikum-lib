import numpy as np;
import pint
import math

from util.structs import CopyManager
from util import myutil;

ureg = pint.UnitRegistry();
ureg.formatter.default_format = "~P"

# I would have *loved* to just extend "Measurement(ufloat)" the ufloat from the uncertainties package
# however the ufloat seems to return a function instead of a py-class, which means extending is not possible
# this is why to add just a bit of custom functions we need to define all operations (__add__, ...) again :(
class Measurement:
    def __init__(self, value, uncertainty, unit=ureg.Unit(""), min_error=0):
         # Überprüfung der Eingaben
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                print(f'Warning! Could not convert value to float: "{value}"')
                value = np.nan

        if not isinstance(value, (float, int, np.int64)):
            print(f"Warning! Value is not a number! Type {type(value)}")
            value = np.nan

        if isinstance(uncertainty, str):
            uncertainty = uncertainty.replace("\"", "")
            if uncertainty.strip().endswith("%"):
                try:
                    percentage = float(uncertainty[:-1])            
                    uncertainty = abs(value) * percentage / 100
                except ValueError:
                    uncertainty = np.nan  # Fallback, falls der Input nicht korrekt ist
            else:
                try:
                    uncertainty = float(uncertainty)
                except ValueError:
                    uncertainty =np.nan

        if not isinstance(uncertainty, (float, int, np.int64)):
            print(f"Warning! Uncertainty is not a number! Type {type(uncertainty)}")
            uncertainty = np.nan;

        # %-Errors could result in 0 values for uncertainties, we can bypass this with a min_error
        uncertainty = np.max([uncertainty, min_error]);

        self.value = value
        self.error = uncertainty

        self.unit = ureg.Unit(unit)
        self.copy = CopyManager(self)

        # displayoptions
        self.display_unit = False;
        self.additional_digits = 0;

    # ==================================================

    def abweichung(self, base):
        return ((self - base) / base).value;

    # ==================================================

    def round_digit(self, digits = 0):
        self.value = myutil.round(self.value, digits)
        self.error = myutil.ceil(self.error, digits)
        return self;

    def round(self, additional_digits = 0):
        exponent = myutil.get_exponent_significant(self.error)
        return self.round_digit(-exponent + additional_digits)
    
    def __round__(self, decimals = 0):
        return self.round_digit(decimals)

    def rint(self):
        return Measurement(np.rint(self.value), np.ceil(self.error))
    
    def to(self, new_unit):
        converted_value = (self.value * self.unit).to(new_unit)
        converted_error = (self.error * self.unit).to(new_unit)
        self.value = converted_value
        self.error = converted_error
        self.unit = new_unit
    
    # ==================================================
    #    math operations
    # ==================================================

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([self + x for x in other]);

        if isinstance(other, (int, float)):
            return self + Measurement(other, 0, ureg.Unit(""))
        
        if isinstance(other, Measurement):        
            value = self.value + other.value;
            error = (self.error**2 + other.error**2)**.5
            if self.unit != other.unit:
                unit = ureg.Unit("");
                print(f"Warning! Addition of different units: [{self.unit}] and [{other.unit}]")
            else:
                unit = self.unit
            return Measurement(value, error, unit);

        raise NotImplementedError(f"Unsupported type for operator +: {type(other)}")
        
    def __radd__(self, other):
        return self.__add__(other);

    def __sub__(self, other):
        return self.__add__(-other);

    def __rsub__(self, other):
        return (-self).__add__(other);

    # ==================================================

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([self * element for element in other]);

        if isinstance(other, (float, int)):
            return self * Measurement(other, 0, ureg.Unit(""))

        if isinstance(other, Measurement):
            value = self.value * other.value;
            error = ((self.error*other.value)**2 + (self.value*other.error)**2)**.5
            unit = self.unit * other.unit
            return Measurement(value, error, unit)

        raise NotImplementedError(f"Unsupported type for operator *: {type(other)}")
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        val = self.value % other
        err = self.error
        return Measurement(val, err)

    # ==================================================

    def __truediv__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([self / x for x in other]);

        if isinstance(other, (float, int, np.int64)):
            return self / Measurement(other, 0, ureg.Unit(""))

        if isinstance(other, Measurement):
            value = self.value / other.value;
            error = ((self.error / other.value)**2 + (self.value * other.error / other.value**2)**2)**.5
            unit = self.unit / other.unit;
            return Measurement(value, error, unit)

        raise NotImplementedError(f"Unsupported type for operator /: {type(other)}")
        
    def __rtruediv__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([x / self for x in other]);

        if isinstance(other, (float, int)):
            return Measurement(other, 0, ureg.Unit("")) / self;

        if isinstance(other, Measurement):
            value = other.value / self.value;
            error = ((other.error / self.value)**2 + (other.value * self.error / self.value**2)**2)**.5
            unit = other.unit / self.unit;
            return Measurement(value, error, unit)

        raise NotImplementedError(f"Unsupported type for reversed-operator /: {type(other)}")

    # ==================================================

    def __pow__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([self**x for x in other]);

        if isinstance(other, (int, float)):
            return self ** Measurement(other, 0, ureg.Unit(""))

        if isinstance(other, Measurement):
            value = self.value**other.value;
            error = ((other.value * self.value**(other.value - 1) * self.error)**2 + (self.value**other.value * np.log(self.value) * other.error)**2)**.5
            if other.unit != ureg.Unit(""):
                print("** Exponent should not have a unit!") 
            unit = self.unit ** other.value;
            return Measurement(value, error, unit);

        raise NotImplementedError(f"Unsupported type for operator **: {type(other)}")
    
    def __rpow__(self, other):
        if isinstance(other, np.ndarray):
            return np.array([x**self for x in other]);

        if isinstance(other, (int, float)):
            return Measurement(other, 0, ureg.Unit("")) ** self

        raise NotImplementedError(f"Unsupported type for operator **: {type(other)}")

    # ==================================================

    def __neg__(self):
        return Measurement(-self.value, self.error, self.unit)

    def __abs__(self):
        return Measurement(abs(self.value), self.error, self.unit)
    
    def exp(self):
        """Ermöglicht np.exp() für Measurement-Objekte"""
        val = np.exp(self.value)
        err = val * self.error  # Fehlerfortpflanzung: Δy = e^x * Δx
        return Measurement(val, err)

    def sqrt(self):
        """Ermöglicht np.exp() für Measurement-Objekte"""
        val = np.sqrt(self.value)
        err = .5 * self.error / np.sqrt(self.value)
        return Measurement(val, err)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy universal functions.
        """
        if method != "__call__":
            return NotImplemented
        
        # Extract the input values and errors
        values = [i.value if isinstance(i, Measurement) else i for i in inputs]
        errors = [i.error if isinstance(i, Measurement) else 0 for i in inputs]

        # Handle specific ufuncs
        match ufunc:
            case np.add:
                val = values[0] + values[1]
                d1 = errors[0]
                d2 = errors[1]
                err = (d1**2 + d2**2)**.5
                return Measurement(val, err)
            case np.subtract:
                val = values[0] - values[1]
                d1 = errors[0]
                d2 = -errors[1]
                err = (d1**2 + d2**2)**.5
                return Measurement(val, err)
            case np.multiply:
                val = values[0] * values[1]
                d1 = (errors[0] * values[1])
                d2 = (values[0] * errors[1])
                err = (d1**2 + d2**2)**.5
                return Measurement(val, err)
            case np.divide:
                val = values[0] / values[1]
                d1 = errors[0] / values[1]
                d2 = - values[0] * errors[1] / values[1]**2
                err = (d1**2 + d2**2)**.5
                return Measurement(val, err)
            case np.sqrt:
                val = np.sqrt(values[0]);
                err = .5 * errors[0] / np.sqrt(values[0])
                return Measurement(val, err)
            case np.sin:
                val = np.sin(values[0])
                err = np.abs(np.cos(values[0]) * errors[0])
                return Measurement(val, err)
            case np.cos:
                val = np.cos(values[0])
                err = np.abs(np.sin(values[0]) * errors[0])
                return Measurement(val, err)
            case np.arcsin:
                val = np.arcsin(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
                return Measurement(val, err)
            case np.arccos:
                val = np.arccos(values[0])
                err = errors[0] / (1 - values[0]**2)**0.5
                return Measurement(val, err)
            case np.tan:
                val = np.tan(values[0])
                err = np.abs((1 / np.cos(values[0])**2) * errors[0])
                return Measurement(val, err)
            case np.exp:
                val = np.exp(values[0]);
                err = np.exp(values[0]) * errors[0];
                return Measurement(val, err);
            case np.log:
                # Propagate the value and error for log(x)
                val = np.log(values[0])
                err = errors[0] / values[0]  # Error propagation formula for log
                return Measurement(val, err)
            case np.log10:
                # Propagate the value and error for log(x)
                val = np.log10(values[0])
                err = errors[0] / values[0] / np.log(10) # Error propagation formula for n-log
                return Measurement(val, err)
            case np.rad2deg:
                val = np.rad2deg(values[0]);
                err = np.rad2deg(errors[0]);
                return Measurement(val, err)
            case np.deg2rad:
                val = np.deg2rad(values[0]);
                err = np.deg2rad(errors[0]);
                return Measurement(val, err)
            case _:
                raise NotImplementedError(f"not handled function: {ufunc}")
        return;

    # ==================================================
    #     Comparison operations
    # ==================================================

    def __lt__(self, other):
        if isinstance(other, (int, float, Measurement)):
            return self.value < other;
        elif isinstance(other, Measurement):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float, Measurement)):
            return self.value <= other;
        elif isinstance(other, Measurement):
            return self.value <= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float, Measurement)):
            return self.value == other;
        elif isinstance(other, Measurement):
            return self.value == other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float, Measurement)):
            return self.value >= other;
        elif isinstance(other, Measurement):
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float, Measurement)):
            return self.value > other;
        elif isinstance(other, Measurement):
            return self.value > other.value
        return NotImplemented
    
    # ==================================================
    #    string operations
    # ==================================================

    @property
    def _value_digit_position(self):
        return myutil.get_exponent_significant(self.value)

    @property
    def _error_digit_position(self):
        return myutil.get_exponent_significant(self.error)

    @property
    def _error_last_digit_position(self):
        return myutil.get_exponent_significant(self.error) - self.additional_digits

    @property
    def _digit_length(self):
        return self._value_digit_position - self._error_digit_position;
    
    def _display_number_as_exponent(self, value, exponent):
        return value / (10 ** exponent);
    
    def to_str_bracket(self, exponent = None) -> str:
        val_start_exp = self._value_digit_position if exponent is None else exponent;
        err_start_exp = self._error_digit_position;
        # if myutil.get_digit_at_exponent(self.error, err_start_exp) == 9 and self.error > 9 * 10**err_start_exp and self.additional_digits is 0:
        #     err_start_exp = err_start_exp + 1;
        err_end_exp = min(err_start_exp - self.additional_digits, val_start_exp);
        length = int(val_start_exp - err_end_exp); 

        if np.isnan(self.value):
            str_val = "-"
        else:
            adjusted_value = self.value / (10 ** val_start_exp)
            str_val = f"{adjusted_value:.{length}f}"

        if np.isnan(self.error) or str_val == "-":
            str_err = ""
        else:
            adjusted_error = self.error / (10 ** err_end_exp)
            adjusted_error = int(myutil.ceil(adjusted_error))
            str_err = f"({adjusted_error:d})"

        str_exp = "" if val_start_exp == 0 else f"e{int(val_start_exp)}"
        return f"{str_val}{str_err}{str_exp}";

    def __str__(self):
        exponent_3n = np.floor(self._value_digit_position / 3) * 3
        return self.to_str_bracket(exponent_3n);
        # # Determine exponents
        # exponent_error = myutil.get_exponent_significant(self.error) - self.additional_digits
        # exponent_value = myutil.get_exponent_significant(self.value)
        # offset = exponent_value - exponent_error
        # if(offset < 0):
        #     exponent_value -= offset;
        #     offset = 0;
        #     print(f"Warning, Error bigger than value {self.value} < {self.error}")

        # # Scientific notation
        # # adjusted_value = self._display_number_as_exponent(self.value, self._value_digit_position);
        # adjusted_value = self._display_number_as_exponent(self.value, exponent_value);
        # value_text = f"{adjusted_value:.{offset}f}"
        # adjusted_error = self._display_number_as_exponent(self.error, self._error_digit_position);
        # error_text = f"{int(myutil.ceil(adjusted_error))}"
        # exponent_text = f"e{exponent_value:+d}" if exponent_value != 0 else ""

        # unit_text = "";
        # if(self.display_unit):
        #     unit_text = " " + str(self.unit);

        # return f"{value_text}({error_text}){exponent_text}{unit_text}"

        # exponent = mymath.get_exponent_closest_3n(self.error);
        # symbol = "+" if exponent > 0 else "-"; # just to get the "+" explicitly
        # exponent_text = f"e{symbol}{abs(exponent)}" if exponent != 0 else "";

        # # unit_text = "[\u00B7]" if (not self.unit or self.unit == "") else f" [{self.unit}]"
        # unit_text = "";

        # # Adjust value and error accordingly
        # adjusted_value = self.value / (10 ** exponent)
        # adjusted_error = self.error / (10 ** exponent)

        # value_text = f"{adjusted_value:>{4 + self.additional_digit}.{self.additional_digit}f}"
        # error_text = f"{adjusted_error:>{2 + self.additional_digit}.{self.additional_digit}f}"
        # return f"({value_text} ± {error_text}){exponent_text}{unit_text}"

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"
    
    def __format__(self, format_spec):
        return format(self.__str__(), format_spec);


# take the value exponnent
# take error exponent

# ex: 1234(5)e-4 -> 123.4(5)e-3 
# value_exp = 0;
# errror_exp = -3;
# width = 3
# width >= 3 -> offset e-3