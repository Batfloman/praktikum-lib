from typing import Optional
import numpy as np;
# import pint

from .. import util
from .measurementBase import MeasurementBase, ConvertibleToFloat
# from structs.measurementBase import MeasurementBase, ConvertibleToFloat

# ureg = pint.UnitRegistry();
# ureg.formatter.default_format = "~P"

class Measurement(MeasurementBase):
    def __init__(self, value: ConvertibleToFloat, uncertainty: ConvertibleToFloat,
                 # unit=ureg.Unit(""),
                 min_error=0):
        super().__init__(value, uncertainty, min_error=min_error)

        # self.unit = ureg.Unit(unit)

        # displayoptions
        self.display_unit = False;
        self.additional_digits = 0;

    # ==================================================

    def abweichung(self, base):
        return ((self - base) / base).value;

    # ==================================================

    def round_digit(self, digits = 0):
        self.value = util.round(self.value, digits)
        self.error = util.ceil(self.error, digits)
        return self;

    def round(self, additional_digits = 0):
        exponent = util.get_exponent_significant(self.error)
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
    #    string operations
    # ==================================================

    @property
    def _value_digit_position(self):
        return util.get_exponent_significant(self.value)

    @property
    def _error_digit_position(self):
        return util.get_exponent_significant(self.error)

    @property
    def _error_last_digit_position(self):
        return util.get_exponent_significant(self.error) - self.additional_digits

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
            adjusted_error = int(util.ceil(adjusted_error))
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

    def save_latex(self, path: str, unit: Optional[str] = None, use_si_prefix: bool = True, print_success_msg: bool = True) -> str:
        from ..tables.latex_table.formatter import format_unit
        from ..tables.validation import ensure_extension

        string = f"{self}"
        if "e" in string:
            num_text = string.split("e")[0]
            exp = int(string.split("e")[1])
            unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
        else:
            num_text = string
            unit_text = format_unit(unit, use_si_prefix=use_si_prefix)

        latex_str = fr"\ensuremath{{{num_text}}}\,{unit_text}"

        # In Datei schreiben
        path = ensure_extension(path, ".tex")

        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        if print_success_msg:
            print(f"{self} has been saved to {path}")

        return latex_str

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"
    
    def __format__(self, format_spec):
        return format(self.__str__(), format_spec);
