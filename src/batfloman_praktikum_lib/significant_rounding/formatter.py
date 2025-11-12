from enum import Enum
from typing import Literal, Union
import re
import numpy as np

from .core import get_sig_digit_position, round_sig, round_sig_fixed

def _get_3n_exponent(uncertainty: float) -> int:
    exponent = np.log10(abs(uncertainty))
    if exponent < 0 and (abs(exponent) % 3) < 1:
        exponent = np.ceil(exponent / 3) * 3
    else:
        exponent = np.floor(exponent / 3) * 3
    return int(exponent)

def _extract_precision(format_spec: str) -> int | None:
    """
    Extract the precision N from a format spec like '.2f', '12.3e', '>.4g', etc.
    Returns None if no precision is specified.
    """
    m = re.search(r'\.(\d+)', format_spec)
    if m:
        return int(m.group(1))
    return None

def _display_plusminus(val: float, err: float, format_spec: str = "") -> str:
    if format_spec.endswith(("e3", "e")):
        decimals = _extract_precision(format_spec)
        if decimals is None:
            decimals = 0
        exponent = _get_3n_exponent(err) if format_spec.endswith("e3") else get_sig_digit_position(err)
        val /= 10**exponent
        err /= 10**exponent
        val, err = round_sig_fixed(val, err, decimals)
        val_str = format(val, f".{decimals}f")
        err_str = format(err, f".{decimals}f")
        exp_str = f"e{int(exponent)}" if exponent != 0 else "";
        return f"({val_str} ± {err_str}){exp_str}"
    
    decimals = -get_sig_digit_position(err)

    if decimals >= 0:
        val_str = format(val, f".{decimals}f")
        err_str = format(err, f".{decimals}f")
    else:
        factor = 10 ** -decimals
        val_str = f"{round(val / factor) * factor:.0f}"
        err_str = f"{round(err / factor) * factor:.0f}"

    return f"{val_str} ± {err_str}"

def _display_parenthesis(val: float, err: float, format_spec: str = "") -> str:
    if format_spec.endswith(("e3", "e")):
        decimals = _extract_precision(format_spec)
        if decimals is None:
            decimals = 0
        exponent = _get_3n_exponent(err) if format_spec.endswith("e3") else get_sig_digit_position(err)
        val /= 10**exponent
        err /= 10**(exponent - decimals)
        val, err = round_sig_fixed(val, err, decimals)
        val_str = format(val, f".{decimals}f")
        err_str = format(err, f".0f")
        exp_str = f"e{int(exponent)}" if exponent != 0 else "";
        return f"{val_str}({err_str}){exp_str}"

    decimals = 0;
    if format_spec.endswith("f"):
        decimals = _extract_precision(format_spec)
        if decimals is None:
            decimals = 0

    sig_digit = min(get_sig_digit_position(err), 0)

    val, err = round_sig_fixed(val, err, -sig_digit + decimals)
    val_str = format(val, f".{-sig_digit + decimals}f")
    err *= 10**(-sig_digit + decimals)
    err_str = format(err, f".0f")

    return f"{val_str}({err_str})"

class UncertaintyNotation(Enum):
    PlusMinus = "pm"     # für ±-Notation, z.B. 1.23 ± 0.01
    Parentheses = "brk" # für Klammer-Notation, z.B. 1.23(1)

def format_measurement(
    value: float,
    uncertainty: float,
    mode: Union[UncertaintyNotation, Literal["pm", "brk"]] = UncertaintyNotation.PlusMinus,
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
