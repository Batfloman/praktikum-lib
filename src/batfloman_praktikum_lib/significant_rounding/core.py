import numpy as np
from typing import Tuple

from .. import util 

def get_sig_digit_position(uncertainty: float) -> int:
    if uncertainty <= 0:
        raise ValueError("Uncertainty must be positive.")
    
    exponent = np.floor(np.log10(uncertainty))
    leading_digit = int(uncertainty / (10 ** exponent))

    # DIN-Norm: 1 oder 2 → zwei signifikante Stellen
    if leading_digit in [1, 2]:
        return int(exponent) - 1
    else:
        return int(exponent)

def round_sig(value: float, uncertainty: float) -> Tuple[float, float]:
    sig_digit_pos = get_sig_digit_position(uncertainty);
    val = util.round(value, -sig_digit_pos)
    err = util.ceil(uncertainty, -sig_digit_pos)
    return (val, err)
