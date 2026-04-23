from typing import Optional, Tuple

import math

from batfloman_praktikum_lib.io.formatters.formatters import custom_format
from ..optionTypes import ValueOptions

def get_exponent(value: float) -> int:
    if value == 0:
        return 0
    return int(math.floor(math.log10(abs(value))))

def normalize(value, uncertainty, options: ValueOptions) -> Tuple[float, float | None]:
    if not options.with_error:
        return value, None

    if uncertainty is None:
        return value, None

    return float(value), float(uncertainty)

def scale(value, uncertainty, options: ValueOptions):
    if options.fixed_exponent is not None:
        exp = options.fixed_exponent
    else:
        exp = get_exponent(float(value))

    value_scaled = value / 10**exp

    if uncertainty is None:
        return value_scaled, None, exp

    uncertainty_scaled = uncertainty / 10**exp
    return value_scaled, uncertainty_scaled, exp

def format_numbers(value, uncertainty, options: ValueOptions):
    v = custom_format(value, options.format_spec)

    if uncertainty is None:
        return v, None

    u = custom_format(uncertainty, options.format_spec)

    return v, u

def get_siunitx_unit(unit: Optional[str] = None, unit_map: Optional[dict[str, str]] = None) -> str | None:
    if unit_map and unit in unit_map:
        return unit_map[unit]
    return unit
