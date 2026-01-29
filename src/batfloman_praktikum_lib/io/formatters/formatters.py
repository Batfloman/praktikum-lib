import re
import numpy as np

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase

from .helpers import get_3n_exponent, extract_precision, get_first_digit_position
from .measurement import custom_format_measurement

type FormattingSupported = int | float | np.integer | np.floating | "MeasurementBase"

def custom_format(
    value: FormattingSupported,
    format_spec: str = "",
) -> str:
    if isinstance(value, MeasurementBase):
        return custom_format_measurement(value, format_spec)

    fval = float(value)

    m = re.search(r"e3n(-?\d+)?", format_spec)
    if m is not None:
        fixed_n = m.group(1)
        exp = 3 * int(fixed_n) if (fixed_n is not None) else get_3n_exponent(fval)
        val = fval / 10**exp
        pre = extract_precision(format_spec)
        if pre is None:
            pre = 6

        exp_str = "" if exp == 0 else f"e{exp:+03d}"
        return f"{val:.{pre}f}{exp_str}"

    m = re.search(r"e(-?\d+)?", format_spec)
    if m is not None:
        if m.group(1) is not None:
            exp = int(m.group(1))
        else:
            exp = get_first_digit_position(fval) or 0

        val = fval / 10**exp
        pre = extract_precision(format_spec)
        if pre is None:
            pre = 6

        exp_str = "" if exp == 0 else f"e{exp:+03d}"
        return f"{val:.{pre}f}{exp_str}"

    return format(fval, format_spec)
