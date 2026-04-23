from typing import Optional, Tuple

import math

from batfloman_praktikum_lib.io.formatters.formatters import custom_format

from ..optionTypes import ValueOptions
from .format_maps import SI_PREFIX_MAP, SIUNITX_UNIT_MAP


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


def get_siunitx_unit(
    unit: Optional[str] = None,
    unit_map: Optional[dict[str, str]] = None,
) -> str | None:
    if unit_map and unit in unit_map:
        return unit_map[unit]
    return unit


def format_unit_body(
    unit: Optional[str],
    *,
    exponent: Optional[int] = None,
    use_si_prefix: bool = True,
) -> str:
    if unit is None or unit == "":
        if exponent:
            return rf"\ensuremath{{10^{exponent}}}"
        return ""

    unit_str = get_siunitx_unit(unit, SIUNITX_UNIT_MAP) or unit

    if not exponent:
        return unit_str

    if use_si_prefix and exponent in SI_PREFIX_MAP:
        prefix = SI_PREFIX_MAP[exponent]
        return f"{prefix}{unit_str}"

    return rf"\ensuremath{{10^{exponent}}}\,\si{{{unit_str}}}"


def format_unit_latex(
    unit: Optional[str],
    *,
    exponent: Optional[int] = None,
    use_si_prefix: bool = True,
) -> str:
    unit_body = format_unit_body(
        unit,
        exponent=exponent,
        use_si_prefix=use_si_prefix,
    )

    if unit is None or unit == "":
        return unit_body

    if exponent and not (use_si_prefix and exponent in SI_PREFIX_MAP):
        return unit_body

    return fr"\si{{{unit_body}}}"
