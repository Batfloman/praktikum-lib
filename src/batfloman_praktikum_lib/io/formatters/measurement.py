from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from .helpers import get_3n_exponent, extract_precision

def custom_format_measurement(value: MeasurementBase, format_spec: str) -> str:
    if "e3" in format_spec:
        exp = get_3n_exponent(value.value)
        val = value / 10**exp
        pre = extract_precision(format_spec) or 0

        exp_str = "" if exp == 0 else f"e{exp}"

        return f"{val:.{pre}f}{exp_str}"

    return ""
