import numpy as np
import re

def get_first_digit_position(val: float) -> int:
    if val == 0:
        raise ValueError("Zero has no leading digit!")
    return int(np.floor(np.log10(abs(val))))

def get_3n_exponent(uncertainty: float) -> int:
    exponent = np.log10(abs(uncertainty))
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
