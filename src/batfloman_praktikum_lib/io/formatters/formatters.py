import numbers
from batfloman_praktikum_lib.io.formatters.measurement import custom_format_measurement
from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from .helpers import get_3n_exponent, extract_precision

type FormattingSupported = numbers.Real | MeasurementBase

def custom_format(
    value: FormattingSupported,
    format_spec: str
) -> str:
    from batfloman_praktikum_lib.structs.measurement import Measurement

    if isinstance(value, Measurement):
        return custom_format_measurement(value, format_spec);

    fval = float(value)

    if "e3" in format_spec:
        exp = get_3n_exponent(fval)
        val = value / 10**exp
        pre = extract_precision(format_spec) or 0

        exp_str = "" if exp == 0 else f"e{exp}"

        return f"{val:.{pre}f}{exp_str}"
    else:
        return format(value, format_spec)
