from enum import Enum

from .core import get_sig_digit_position, round_sig

class UncertaintyNotation(Enum):
    PlusMinus = "PlusMinus"     # für ±-Notation, z.B. 1.23 ± 0.01
    Parentheses = "Parentheses" # für Klammer-Notation, z.B. 1.23(1)

def _display_plusminus(value: float, uncertainty: float) -> str:
    val, err = round_sig(value, uncertainty)
    decimals = -get_sig_digit_position(uncertainty)
    return f"{val:.{decimals}f} ± {err:.{decimals}f}"

def _display_parenthesis(value: float, uncertainty: float) -> str:
    val, err = round_sig(value, uncertainty)
    decimals = -get_sig_digit_position(uncertainty)
    err_str = f"{int(err * 10**decimals)}"
    return f"{val:.{decimals}f}({err_str})"

def format_measurement(value: float, uncertainty: float, mode: UncertaintyNotation = UncertaintyNotation.PlusMinus) -> str:
    match mode:
        case UncertaintyNotation.PlusMinus:
            return _display_plusminus(value, uncertainty)
        case UncertaintyNotation.Parentheses:
            return _display_parenthesis(value, uncertainty)
        case _:
            raise ValueError(f"Unknown uncertainty notation: {mode}")

