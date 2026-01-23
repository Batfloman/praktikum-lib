from typing import Literal
from enum import Enum

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from batfloman_praktikum_lib.significant_rounding import get_sig_digit_position, round_sig_fixed
from .helpers import get_3n_exponent, extract_precision, get_first_digit_position

class UncertaintyNotation(Enum):
    PlusMinus = "pm"     # für ±-Notation, z.B. 1.23 ± 0.01
    Parentheses = "brk" # für Klammer-Notation, z.B. 1.23(1)

def custom_format_measurement(value: MeasurementBase, format_spec: str) -> str:
    notation = UncertaintyNotation.Parentheses

    if UncertaintyNotation.PlusMinus.value in format_spec:
        notation = UncertaintyNotation.PlusMinus
        format_spec = format_spec.replace(UncertaintyNotation.PlusMinus.value, ""); 
    elif UncertaintyNotation.Parentheses.value in format_spec:
        notation = UncertaintyNotation.Parentheses
        format_spec = format_spec.replace(UncertaintyNotation.Parentheses.value, ""); 

    return format_measurement(
        value.value,
        value.error,
        mode = notation,
        format_spec = format_spec,
    )

def format_measurement(
    value: float,
    uncertainty: float,
    mode: UncertaintyNotation | Literal["pm", "brk"] = UncertaintyNotation.PlusMinus,
    format_spec: str = ""
) -> str:
    """
    Format a value ± uncertainty with optional numeric formatting.
    - mode: 'pm' or 'brk'
    - format_spec: e.g. '.2f', '.3e'
    """
    if isinstance(mode, str):
        mode = UncertaintyNotation(mode)

    match mode:
        case UncertaintyNotation.PlusMinus:
            return _display_plusminus(value, uncertainty, format_spec)
        case UncertaintyNotation.Parentheses:
            return _display_parenthesis(value, uncertainty, format_spec)
        case _:
            raise ValueError(f"Unknown uncertainty notation: {mode}")

def _display_plusminus_exponent(val: float, err: float, exponent: int, decimals: int = 0) -> str:
    val /= 10**exponent
    err /= 10**exponent
    val, err = round_sig_fixed(val, err, decimals)
    val_str = format(val, f".{decimals}f")
    err_str = format(err, f".{decimals}f")
    exp_str = f"e{int(exponent)}" if exponent != 0 else "";
    return f"({val_str} ± {err_str}){exp_str}"

def _display_plusminus_float(val: float, err: float, decimals: int = 0) -> str:
    if decimals >= 0:
        val_str = format(val, f".{decimals}f")
        err_str = format(err, f".{decimals}f")
    else:
        factor = 10 ** -decimals
        val_str = f"{round(val / factor) * factor:.0f}"
        err_str = f"{round(err / factor) * factor:.0f}"

    return f"{val_str} ± {err_str}"

def _display_plusminus(val: float, err: float, format_spec: str = "") -> str:
    decimals = extract_precision(format_spec) or 0;

    if format_spec.endswith(("e3", "e")):
        exponent = get_3n_exponent(err) if format_spec.endswith("e3") else get_sig_digit_position(err)
        return _display_plusminus_exponent(val, err, exponent, decimals);
    if format_spec.endswith("f"):
        return _display_plusminus_float(val, err, decimals)
    
    exponent = get_sig_digit_position(err);
    if abs(exponent) < 6:
        decimals = max(0, -get_sig_digit_position(err))
        return _display_plusminus_float(val, err, decimals)
    else:
        return _display_plusminus_exponent(val, err, exponent, 0)

# ==================================================
# parenthesis notation

def _display_parenthesis_exponent(val: float, err: float, exponent: int, decimals: int = 0) -> str:
    val /= 10**exponent
    err /= 10**(exponent - decimals)

    val, _ = round_sig_fixed(val, 0, decimals)
    _, err = round_sig_fixed(0, err, decimals=0)

    val_str = format(val, f".{decimals}f")
    err_str = format(err, f".0f")
    exp_str = f"e{int(exponent)}" if exponent != 0 else "";
    return f"{val_str}({err_str}){exp_str}"

def _display_parenthesis_float(val: float, err: float, decimals: int = 0) -> str:
    sig_digit = min(get_sig_digit_position(err), 0)

    val, err = round_sig_fixed(val, err, -sig_digit + decimals)
    val_str = format(val, f".{-sig_digit + decimals}f")
    err *= 10**(-sig_digit + decimals)
    err_str = format(err, f".0f")

    return f"{val_str}({err_str})"

def _display_parenthesis(val: float, err: float, format_spec: str = "") -> str:
    decimals = extract_precision(format_spec) or 0;

    if "e3" in format_spec:
        exponent = get_3n_exponent(val)
        err_exp = get_sig_digit_position(err)
        offset = max(exponent - err_exp, 0)
        if err_exp - exponent > 0:
            exponent = get_3n_exponent(err)
        return _display_parenthesis_exponent(val, err, exponent, offset + decimals)

    if "f" in format_spec:
        return _display_parenthesis_float(val, err, decimals);

    if "e" in format_spec:
        err_exp = get_sig_digit_position(err)
        exponent = err_exp if (tmp := get_first_digit_position(val)) is None else tmp
        offset = max(exponent - err_exp, 0)

        return _display_parenthesis_exponent(val, err, exponent, offset + decimals)

    # if "g" in format_spec:
    # default
    err_exp = get_sig_digit_position(err)
    exponent = err_exp if (tmp := get_first_digit_position(val)) is None else tmp
    offset = max(exponent - err_exp, 0)

    if -3 <= exponent < 6 and err_exp < exponent:
        return _display_parenthesis_float(val, err, decimals)
    else:
        return _display_parenthesis_exponent(val, err, exponent, offset + decimals);

