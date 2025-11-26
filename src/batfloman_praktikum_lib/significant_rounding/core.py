import numpy as np
from typing import Tuple

from .. import util 

def get_sig_digit_position(uncertainty: float) -> int:
    if uncertainty <= 0:
        raise ValueError("Uncertainty must be positive.")

    exponent = int(np.floor(np.log10(uncertainty)))
    leading_digit = int((uncertainty / (10 ** exponent) + 1e-12))

    # DIN-Norm: leading digit 1 or 2 → two significant digits
    if leading_digit in [1, 2]:
        return exponent - 1
    else:
        return exponent

def round_sig(value: float, uncertainty: float) -> Tuple[float, float]:
    sig_digit_pos = get_sig_digit_position(uncertainty);
    val = util.round(value, -sig_digit_pos)
    err = util.ceil(uncertainty, -sig_digit_pos)
    return (val, err)

def round_sig_fixed(value: float, uncertainty: float, decimals: int) -> Tuple[float, float]:
    val = util.round(value, decimals)
    err = util.ceil(uncertainty, decimals)
    return (val, err)
