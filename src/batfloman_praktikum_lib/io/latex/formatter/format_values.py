from typing import Optional

import numbers

from ..optionTypes import ValueOptions
from ._helper import get_siunitx_unit
from .format_maps import SI_PREFIX_MAP, SIUNITX_UNIT_MAP
from batfloman_praktikum_lib.io.formatters import custom_format

def format_number_latex_str(
    value,
    *,
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    format_spec: str = "",
) -> str:
    if unit is None:
        if fixed_exponent is None:
            formatted = custom_format(value, format_spec)
            return rf"\num{{{formatted}}}"

        val = value / 10**fixed_exponent
        formatted = custom_format(val, format_spec)
        return rf"\num[scientific-notation=fixed, fixed-exponent={fixed_exponent}]{{{formatted}e{fixed_exponent}}}"

    unit_str = get_siunitx_unit(unit, SIUNITX_UNIT_MAP) or unit

    if fixed_exponent is not None:
        val = value / 10**fixed_exponent
        formatted = custom_format(val, format_spec)
        if "e" in formatted:
            raise ValueError("Unexpected scientific notation in formatted value")

        if use_si_prefix and fixed_exponent in SI_PREFIX_MAP:
            unit_text = f"{SI_PREFIX_MAP[fixed_exponent]}{unit_str}"
            return rf"\SI[scientific-notation=fixed]{{{formatted}}}{{{unit_text}}}"

        return rf"\SI[scientific-notation=fixed, fixed-exponent={fixed_exponent}]{{{formatted}e{fixed_exponent}}}{{{unit_str}}}"

    formatted = custom_format(value, format_spec)

    if "e" not in formatted:
        return rf"\SI{{{formatted}}}{{{unit_str}}}"

    _, exp_str = formatted.split("e")
    exp_val = int(exp_str)
    num_val = value / 10**exp_val
    formatted = custom_format(num_val)

    if use_si_prefix and exp_val in SI_PREFIX_MAP:
        unit_text = f"{SI_PREFIX_MAP[exp_val]}{unit_str}"
        return rf"\SI{{{formatted}}}{{{unit_text}}}"

    return rf"\SI{{{formatted}e{exp_val}}}{{{unit_str}}}"

def format_value(
    value: numbers.Real,
    uncertainty: numbers.Real | None = None,
    *,
    options: Optional[ValueOptions] = None,
) -> str:
    options = options or ValueOptions()

    if uncertainty is None or not options.with_error:
        obj = value
    else:
        from batfloman_praktikum_lib.structs.measurement import Measurement
        obj = Measurement(value, uncertainty)

    return format_number_latex_str(
        obj,
        unit=options.unit,
        use_si_prefix=options.use_si_prefix,
        fixed_exponent=options.fixed_exponent,
        format_spec=options.format_spec,
    )
