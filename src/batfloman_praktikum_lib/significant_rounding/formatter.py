from enum import Enum
from typing import Literal, Union

from .core import get_sig_digit_position, round_sig

def _display_plusminus(value: float, uncertainty: float) -> str:
    val, err = round_sig(value, uncertainty)
    decimals = -get_sig_digit_position(uncertainty)
    return f"{val:.{decimals}f} ± {err:.{decimals}f}"

def _display_parenthesis(value: float, uncertainty: float) -> str:
    val, err = round_sig(value, uncertainty)
    decimals = -get_sig_digit_position(uncertainty)
    err_str = f"{int(err * 10**decimals)}"
    return f"{val:.{decimals}f}({err_str})"

class UncertaintyNotation(Enum):
    PlusMinus = "pm"     # für ±-Notation, z.B. 1.23 ± 0.01
    Parentheses = "brk" # für Klammer-Notation, z.B. 1.23(1)

def format_measurement(value: float, uncertainty: float, mode: Union[UncertaintyNotation, Literal["pm", "brk"]] = UncertaintyNotation.PlusMinus) -> str:
    if isinstance(mode, str):
        mode = UncertaintyNotation(mode)

    match mode:
        case UncertaintyNotation.PlusMinus:
            return _display_plusminus(value, uncertainty)
        case UncertaintyNotation.Parentheses:
            return _display_parenthesis(value, uncertainty)
        case _:
            raise ValueError(f"Unknown uncertainty notation: {mode}")

