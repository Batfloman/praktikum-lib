from typing import NamedTuple, Optional, Union, Literal
import numpy as np;
import json
# import pint

from .. import util
from .measurementBase import MeasurementBase, ConvertibleToFloat, _get_value_and_error
# from structs.measurementBase import MeasurementBase, ConvertibleToFloat
from ..significant_rounding.formatter import format_measurement, UncertaintyNotation

# ureg = pint.UnitRegistry();
# ureg.formatter.default_format = "~P"

class Abweichung(NamedTuple):
    sigma: float;
    percent: float;
    ratio: float;

class Measurement(MeasurementBase):
    def __init__(self, value: ConvertibleToFloat, uncertainty: ConvertibleToFloat,
                 # unit=ureg.Unit(""),
                 min_error=0):
        super().__init__(value, uncertainty, min_error=min_error)

    # ==================================================

    def abweichung(self, base) -> Abweichung:
        other_val, other_err = _get_value_and_error(base)
        if other_val == 0:
            raise ValueError("Cannot calculate Abweichung to 0")

        ratio = self.value / other_val
        combined_error = (self.error**2 + other_err**2)**0.5
        return Abweichung(
            ratio=ratio,
            percent=(ratio - 1) * 100,
            sigma=abs(self.value - other_val) / combined_error
        )

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
    
    # def save_latex(self, path: str, unit: Optional[str] = None, use_si_prefix: bool = True, print_success_msg: bool = True) -> str:
    #     from ..tables.latex_table.formatter import format_unit
    #     from ..tables.validation import ensure_extension
    #
    #     string = f"{self}"
    #     if "e" in string:
    #         num_text = string.split("e")[0]
    #         exp = int(string.split("e")[1])
    #         unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
    #     else:
    #         num_text = string
    #         unit_text = format_unit(unit, use_si_prefix=use_si_prefix)
    #
    #     latex_str = fr"\ensuremath{{{num_text}}}\,{unit_text}"
    #
    #     # In Datei schreiben
    #     path = ensure_extension(path, ".tex")
    #
    #     with open(path, "w", encoding="utf-8") as f:
    #         f.write(latex_str)
    #     if print_success_msg:
    #         print(f"{self} has been saved to {path}")
    #
    #     return latex_str

    def save_latex(
        self,
        path: str,
        unit: Optional[str] = None,
        use_si_prefix: bool = True,
        print_success_msg: bool = True,
        mode: Union[UncertaintyNotation, Literal["pm", "brk"]] = UncertaintyNotation.Parentheses,
        format_spec: str = "",
        with_error: bool = True,
    ) -> str:
        """
        Saves a LaTeX string representing the measurement with its uncertainty.
        Supports both ± and bracket notation via `mode`.
        """
        from ..tables.latex_table.formatter import _format_unit as format_unit
        from ..tables.validation import ensure_extension

        # 🔹 Use the new formatter instead of str(self)
        formatted = format_measurement(self.value, self.error, mode=mode, format_spec=format_spec)

        # 🔹 Handle scientific notation (e.g. 1.23e-3)
        if "e" in formatted:
            num_text, exp_str = formatted.split("e")
            exp = int(exp_str)
            unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
        else:
            num_text = formatted
            unit_text = format_unit(unit, use_si_prefix=use_si_prefix)

        if not with_error:
            if "±" in num_text:
                num_text = num_text.split("±")[0]
                if "(" in num_text:
                    num_text = num_text.split("(")[1]
            elif "(" in num_text:
                num_text = num_text.split("(")[0]

        latex_str = fr"\ensuremath{{{num_text}}}\,{unit_text}"

        # 🔹 Write to file
        path = ensure_extension(path, ".tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_str)

        if print_success_msg:
            print(f"Saved [{formatted}] to {path}")

        return latex_str

    # ==================================================

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "error": self.error
        }

    @staticmethod
    def from_dict(d):
        return Measurement(d["value"], d["error"])

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(json_str: str):
        return Measurement.from_dict(json.loads(json_str))

    def save_json(self, path: str, indent: Optional[int] = 3):
        from ..tables.validation import ensure_extension
        path = ensure_extension(path, ".json")

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @staticmethod
    def load_json(path: str):
        from ..tables.validation import ensure_extension
        path = ensure_extension(path, ".json")

        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
        return Measurement.from_json(json_str)

    # ==================================================

    def __str__(self):
        return format_measurement(self.value, self.error, mode="brk");

    def __repr__(self):
        return f"Measurement(value={self.value}, error={self.error})"

    def __format__(self, format_spec):
        if format_spec in ("pm", "brk"):
            return format_measurement(self.value, self.error, mode=format_spec)
        # numeric formats like .2f, .3e, etc.
        return format_measurement(self.value, self.error, mode="brk", format_spec=format_spec)

