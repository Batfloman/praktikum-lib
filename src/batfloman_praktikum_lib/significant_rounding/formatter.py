from enum import Enum
from typing import Literal, Union
import re
import numpy as np

from .core import get_sig_digit_position, round_sig, round_sig_fixed

def _get_first_digit_position(val: float) -> int:
    if val == 0:
        raise ValueError("Zero has no leading digit!")
    return int(np.floor(np.log10(abs(val))))

def _get_3n_exponent(uncertainty: float) -> int:
    exponent = np.log10(abs(uncertainty))
    if exponent < 0 and (abs(exponent) % 3) < 1.7:
        exponent = np.ceil(exponent / 3) * 3
    elif exponent > 0 and (abs(exponent) % 3) > 2.5:
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

def _diplay_plusminus_exponent(val: float, err: float, exponent: int, decimals: int = 0) -> str:
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
    decimals = _extract_precision(format_spec) or 0;

    if format_spec.endswith(("e3", "e")):
        exponent = _get_3n_exponent(err) if format_spec.endswith("e3") else get_sig_digit_position(err)
        return _diplay_plusminus_exponent(val, err, exponent, decimals);
    if format_spec.endswith("f"):
        return _display_plusminus_float(val, err, decimals)
    
    exponent = get_sig_digit_position(err);
    if abs(exponent) < 6:
        return _display_plusminus_float(val, err, 0)
    else:
        return _diplay_plusminus_exponent(val, err, exponent, 0)

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
    decimals = _extract_precision(format_spec) or 0;

    if format_spec.endswith("e3"):
        exponent = _get_3n_exponent(err)
        return _display_parenthesis_exponent(val, err, exponent, decimals)
    if format_spec.endswith(("e3", "e")):
        exponent = get_sig_digit_position(err)
        try:
            first = _get_first_digit_position(val)
        except ValueError as e:
            first = None
        offset = max(first - exponent, 0) if first else 0

        return _display_parenthesis_exponent(val, err, exponent + offset, decimals + offset)
    if format_spec.endswith("f"):
        return _display_parenthesis_float(val, err, decimals);

    exponent = get_sig_digit_position(err);
    if abs(exponent) < 6:
        return _display_parenthesis_float(val, err, 0)
    else:
        try:
            first = _get_first_digit_position(val)
        except ValueError as e:
            first = None
        offset = max(first - exponent, 0) if first else 0

        return _display_parenthesis_exponent(val, err, exponent + offset, offset)

# def _display_parenthesis(val: float, err: float, format_spec: str = "") -> str:
#     if format_spec.endswith(("e3", "e")):
#         decimals = _extract_precision(format_spec)
#         if decimals is None:
#             decimals = 0
#         exponent = _get_3n_exponent(err) if format_spec.endswith("e3") else get_sig_digit_position(err)
#         val /= 10**exponent
#         err /= 10**(exponent - decimals)
#
#         val, _ = round_sig_fixed(val, 0, decimals)
#         _, err = round_sig_fixed(0, err, decimals=0)
#
#         val_str = format(val, f".{decimals}f")
#         err_str = format(err, f".0f")
#         exp_str = f"e{int(exponent)}" if exponent != 0 else "";
#         return f"{val_str}({err_str}){exp_str}"
#
#     decimals = 0;
#     if format_spec.endswith("f"):
#         decimals = _extract_precision(format_spec)
#         if decimals is None:
#             decimals = 0
#
#     sig_digit = min(get_sig_digit_position(err), 0)
#
#     val, err = round_sig_fixed(val, err, -sig_digit + decimals)
#     val_str = format(val, f".{-sig_digit + decimals}f")
#     err *= 10**(-sig_digit + decimals)
#     err_str = format(err, f".0f")
#
#     return f"{val_str}({err_str})"

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
