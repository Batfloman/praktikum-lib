from typing import Optional
import numpy as np;
# import pint

from .. import util
from .measurementBase import MeasurementBase, ConvertibleToFloat
from ..significant_rounding import format_measurement
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

    def __str__(self):
        return format_measurement(self.value, self.error, mode="brk");

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        if format_spec in ("pm", "brk"):
            return format_measurement(self.value, self.error, mode=format_spec)
        # numeric formats like .2f, .3e, etc.
        return format_measurement(self.value, self.error, mode="brk", format_spec=format_spec)

