from typing import Optional
from typing import Literal

import numbers

from ..optionTypes import ValueOptions, normalize_value_options
from ._number_helper import format_text_unit_suffix, format_unit_body, resolve_unit_mode
from .format_maps import SI_PREFIX_MAP
from batfloman_praktikum_lib.io.formatters import custom_format

def format_number_latex_str(
    value,
    *,
    unit: Optional[str] = None,
    unit_mode: Literal["auto", "siunitx", "text"] = "auto",
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

    resolved_unit_mode = resolve_unit_mode(unit, unit_mode)

    if resolved_unit_mode == "text":
        if fixed_exponent is not None and use_si_prefix and fixed_exponent in SI_PREFIX_MAP:
            raise ValueError("SI prefixes are not supported for text-mode units.")

        if fixed_exponent is None:
            formatted = custom_format(value, format_spec)
            return rf"\num{{{formatted}}}\,{format_text_unit_suffix(unit)}"

        val = value / 10**fixed_exponent
        formatted = custom_format(val, format_spec)
        return rf"\num[scientific-notation=fixed, fixed-exponent={fixed_exponent}]{{{formatted}e{fixed_exponent}}}\,{format_text_unit_suffix(unit)}"

    unit_str = format_unit_body(unit)

    if fixed_exponent is not None:
        val = value / 10**fixed_exponent
        formatted = custom_format(val, format_spec)
        if "e" in formatted:
            raise ValueError("Unexpected scientific notation in formatted value")

        if use_si_prefix and fixed_exponent in SI_PREFIX_MAP:
            unit_text = format_unit_body(unit, exponent=fixed_exponent, use_si_prefix=use_si_prefix)
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
        unit_text = format_unit_body(unit, exponent=exp_val, use_si_prefix=use_si_prefix)
        return rf"\SI{{{formatted}}}{{{unit_text}}}"

    return rf"\SI{{{formatted}e{exp_val}}}{{{unit_str}}}"

def format_value(
    value: numbers.Real,
    uncertainty: numbers.Real | None = None,
    *,
    options: Optional[ValueOptions] = None,
) -> str:
    options = normalize_value_options(options)

    if uncertainty is None or not options["with_error"]:
        obj = value
    else:
        from batfloman_praktikum_lib.structs.measurement import Measurement
        obj = Measurement(value, uncertainty)

    return format_number_latex_str(
        obj,
        unit=options["unit"],
        unit_mode=options["unit_mode"],
        use_si_prefix=options["use_si_prefix"],
        fixed_exponent=options["fixed_exponent"],
        format_spec=options["format_spec"],
    )
