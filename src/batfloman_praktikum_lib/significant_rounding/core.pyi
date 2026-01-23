from typing import Tuple

def get_sig_digit_position(uncertainty: float) -> int:
    """
    Return the significant digit of an (uncertainty) value using the DIN-Norm
    leading digit is 1 or 2 → two significant digits
    """
    ...

def round_sig(value: float, uncertainty: float) -> Tuple[float, float]:
    ...

def round_sig_fixed(value: float, uncertainty: float, decimals: int) -> Tuple[float, float]:
    ...
