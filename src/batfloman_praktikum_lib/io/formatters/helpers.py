import numpy as np
import re

def get_first_digit_position(val: float) -> int | None:
    if val == 0:
        return None
    if np.isnan(val) or np.isinf(val):
        return None 

    return int(np.floor(np.log10(abs(val))))

def get_3n_exponent(x: float) -> int:
    if x == 0.0:
        return 0

    exponent = np.log10(abs(x))
    return int(np.floor(exponent / 3) * 3)

    # if exponent < 0 and (abs(exponent) % 3) < 1.7:
    #     exponent = np.floor(exponent / 3) * 3
    # elif exponent > 0 and (abs(exponent) % 3) > 2.5:
    #     exponent = np.ceil(exponent / 3) * 3
    # else:
    #     exponent = np.floor(exponent / 3) * 3
    # return int(exponent)

def extract_precision(format_spec: str) -> int | None:
    """
    Extract the precision N from a format spec like '.2f', '12.3e', '>.4g', etc.
    Returns None if no precision is specified.
    """
    m = re.search(r'\.(\d+)', format_spec)
    if m:
        return int(m.group(1))
    return None
