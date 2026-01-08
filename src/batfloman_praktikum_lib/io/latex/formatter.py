from typing import Optional, cast
import numpy as np
import numbers
import re

from batfloman_praktikum_lib.structs.measurement import Measurement
from .format_maps import SIUNITX_UNIT_MAP, SI_PREFIX_MAP
from ..table_metadata import DEFAULT_ALIGNMENT, ALIGNMENT_VALUES, TableColumnMetadata, TableMetadataManager, normalize_metadata

def get_siunitx_unit(unit: Optional[str] = None) -> str | None:
    if unit in SIUNITX_UNIT_MAP:
        return SIUNITX_UNIT_MAP[unit]
    return None

def _format_unit(
    unit: str | None, 
    *,
    exponent: Optional[int] = None, 
    use_si_prefix: bool = True
) -> str:
    r"""
    Returns a string suitable for siunitx: \si{unit}.
    """
    if unit is None or unit == "":
        if exponent:
            return rf"\ensuremath{{10^{exponent}}}"
        else:
            return ""

    unit_str = get_siunitx_unit(unit) or unit

    if not exponent:
        return fr"\si{{{unit_str}}}"

    if use_si_prefix and (exponent in SI_PREFIX_MAP):
        prefix = SI_PREFIX_MAP[exponent]
        return rf"\si{{{prefix}{unit_str}}}"
    else:
        return rf"\ensuremath{{10^{exponent}}}\,\si{{{unit_str}}}"

def format_number_latex_str(
    value: numbers.Real | Measurement,
    *,
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    format_spec: str = "",
) -> str:
    if unit is None:
        if fixed_exponent is None:
            formatted = format(value, format_spec); 

            return fr"\num{{{formatted}}}"
        else:
            val = value / 10**fixed_exponent
            formatted = format(val, format_spec)

            return fr"\num[scientific-notation=fixed, fixed-exponent={fixed_exponent}]{{{formatted}e{fixed_exponent}}}"
    else:
        if fixed_exponent is not None:
            val = value / 10**fixed_exponent
            formatted = format(val, format_spec)
            if "e" in formatted:
                raise ValueError("Unexpected scientific notation in formatted value")
            
            unit_str = get_siunitx_unit(unit) or unit

            if (use_si_prefix) and (fixed_exponent in SI_PREFIX_MAP):
                unit_text = f"{SI_PREFIX_MAP[fixed_exponent]}{unit_str}"
                return fr"\SI[scientific-notation=fixed]{{{formatted}}}{{{unit_text}}}"
            else:
                return fr"\SI[scientific-notation=fixed, fixed-exponent={fixed_exponent}]{{{formatted}e{fixed_exponent}}}{{{unit_str}}}"
        else:
            formatted = format(value, format_spec)
            unit_str = get_siunitx_unit(unit) or unit

            if "e" not in formatted:
                return fr"\SI{{{formatted}}}{{{unit_str}}}"
            else:
                _, exp_str = formatted.split("e")
                exp_val = int(exp_str)
                num_val = value / 10**exp_val
                formatted = format(num_val)

                if (use_si_prefix) and (exp_val in SI_PREFIX_MAP):
                    unit_text = f"{SI_PREFIX_MAP[exp_val]}{unit_str}"
                    return fr"\SI{{{formatted}}}{{{unit_text}}}"
                else:
                    return fr"\SI{{{formatted}e{exp_val}}}{{{unit_str}}}"

# ==================================================
# tables
# ==================================================

def _get_single_column_format(
    index: str, 
    metadata: Optional[TableColumnMetadata]
) -> str:
    if not metadata:
        return DEFAULT_ALIGNMENT

    s = ""

    if getattr(metadata, "left_border", False):
        s += "|"

    alignment = getattr(metadata, "alignment", DEFAULT_ALIGNMENT)
    if alignment not in ALIGNMENT_VALUES:
        raise ValueError(f"Invalid alignment '{alignment}' for index `{index}`. Must be one of {ALIGNMENT_VALUES}.")
    s += alignment

    if getattr(metadata, "right_border", False):
        s += "|"

    return s

def get_column_format(
    indices: list[str],
    metadata_manager: Optional[TableMetadataManager] = None
) -> str:
    if not metadata_manager:
        return DEFAULT_ALIGNMENT * len(indices)
    individual_formats = [_get_single_column_format(i, metadata_manager.get_metadata(i)) for i in indices]
    return "".join(individual_formats)

def _format_symbol(name: str) -> str:
    # crude heuristic: if it contains `_` or `^` or is a single letter → math mode
    pattern = r"^[A-Za-z](?:([_^](?:\{[^{}]+\}|[A-Za-z0-9]+)))*"
    if re.match(pattern, name):
        return f"${name}$"
    return name

def format_table_header(
    index: str,
    metadata: TableColumnMetadata, 
    *, 
    sep=" in "
) -> str:
    metadata = normalize_metadata(metadata);
    name = metadata.name or _format_symbol(index)

    # Determine unit formatting
    unit = metadata.unit
    exponent = metadata.display_exponent
    use_si = True if (metadata.use_si_prefix is None) else metadata.use_si_prefix

    # Format unit string (empty string if no unit)
    unit_text = _format_unit(unit, exponent=exponent, use_si_prefix=use_si)

    if not unit_text:
        return name;
    else:
        return f"{name}{sep}{unit_text}"

def format_table_value(
    value: numbers.Real | str | np.str_ | Measurement, 
    metadata: TableColumnMetadata,
) -> str:
    metadata = normalize_metadata(metadata);

    # if not convertable, just use the text
    val: float = np.nan
    if isinstance(value, (str, np.str_)):
        try: 
            val = float(value)
        except:
            return value

    if np.isnan(val):
        return "NaN"

    offset_exp = metadata.display_exponent or 0
    val /= 10**offset_exp

    return format_number_latex_str(
        cast(numbers.Real, val), 
        format_spec = metadata.format_spec or ""
    )
