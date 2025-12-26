from ..metadata import ColumnMetadata
from ...structs.measurement import Measurement

from typing import List, Union, Optional
import numpy as np
import re

from ..metadata import MetadataManager, DEFAULT_ALIGNMENT, ALIGNMENT, ALIGNMENT_VALUES
# typing alias
CellValue = Union[float, str, Measurement]

# ==================================================
# helper

SI_PREFIXES = {
     9: r"\giga",  # giga
     6: r"\mega",  # mega
     3: r"\kilo",  # kilo
    -3: r"\milli",  # milli
    -6: r"\micro ",  # micro (space important so that latex \textmu[--Unit--] does not result in an unknown command!)
    # -6: r"\textmu ",  # micro (space important so that latex \textmu[--Unit--] does not result in an unknown command!)
    -9: r"\nano",  # nano
    -12: r"\pico", # pico
    -15: r"\femto", # femto
}

def _format_symbol(name: str) -> str:
    # crude heuristic: if it contains `_` or `^` or is a single letter → math mode
    pattern = r"^[A-Za-z](?:([_^](?:\{[^{}]+\}|[A-Za-z0-9]+)))*"
    if re.match(pattern, name):
        return f"${name}$"
    return name

def _format_exponent(exponent: int | None, use_si_prefix: bool = True) -> str:
    if exponent in (None, 0):
        return "";
    
    return SI_PREFIXES[exponent] if (use_si_prefix and exponent in SI_PREFIXES) else fr"\ensuremath{{ 10^{{{exponent}}} }}";

def _format_unit(
    unit: str | None, 
    exponent: Optional[int] = None, 
    use_si_prefix: bool = True
) -> str:
    r"""
    Returns a string suitable for siunitx: \si{unit}.
    """
    if unit is None or unit == "":
        if not exponent:
            return ""
        else:
            return fr"\ensuremath{{ 10^{{ {exponent} }} }}"

    if not exponent:
        return fr"\si{{ {unit} }}"
    
    if use_si_prefix and exponent in SI_PREFIXES:
        prefix = SI_PREFIXES[exponent]
        return fr"\si {{ {prefix} {unit} }}"
    else:
        return fr"\ensuremath{{ 10^{{{exponent}}} }}\,\si {{ {unit} }}"

# ==================================================

def format_header(index: str, metadata: ColumnMetadata) -> str:
    # Use the name from metadata or fallback to a formatted index
    name = metadata.name or _format_symbol(index)

    # Determine unit formatting
    unit = metadata.unit
    exponent = metadata.display_exponent
    use_si = True if (metadata.use_si_prefix is None) else metadata.use_si_prefix

    # Format unit string (empty string if no unit)
    unit_text = _format_unit(unit, exponent=exponent, use_si_prefix=use_si)
    if unit_text:
        unit_text = f" in {unit_text}"

    return rf"{name}{unit_text}"

# ==================================================

def get_single_column_format(
    index: str, metadata: Optional[ColumnMetadata]
) -> str:
    if not metadata:
        return DEFAULT_ALIGNMENT

    s = ""

    if getattr(metadata, "left_border", False):
        s += "|"

    alignment = getattr(metadata, "alignment", DEFAULT_ALIGNMENT)
    if alignment not in ALIGNMENT_VALUES:
        print(f"Warning: Invalid alignment '{alignment}'. Must be one of {ALIGNMENT_VALUES}.")
        alignment = DEFAULT_ALIGNMENT
    s += alignment

    if getattr(metadata, "right_border", False):
        s += "|"

    return s

def get_column_format(
    indices = list[str],
    metadata_manager: Optional[MetadataManager] = None
) -> str:
    if not metadata_manager:
        return DEFAULT_ALIGNMENT * len(indices)
    return "".join(get_single_column_format(i, metadata_manager.get_metadata(i)) for i in indices)

# ==================================================

def format_value(
    val: Union[int, float, np.integer, np.floating, Measurement, str, np.str_],
    metadata: ColumnMetadata
) -> str:
    """
    Format a single cell value according to the metadata.

    - Applies exponent scaling, SI prefix, and numeric formatting.
    - Returns a LaTeX-ready string.
    """
    if isinstance(val, (str, np.str_)):
        try: 
            val = float(val)
        except:
            return val

    if np.isnan(val):
        return "NaN"

    offset_exp = metadata.display_exponent or 0
    val /= 10**offset_exp

    if metadata.format_spec:
        return f"\\num{{{val:{metadata.format_spec}}}}"

    return f"\\num{{{val}}}"

# def format_value(
#     val: Union[float, str, Measurement],
#     display_exponent: Optional[int] = None,
#     force_exponent: bool = False
# ) -> Union[float, str, Measurement]:
#     if str(val).strip() == "-":
#         return "-"
#     try:
#         num_val = float(val) if isinstance(val, str) else val
#     except:
#         return val
#     shifted_val = num_val / 10**display_exponent if display_exponent else num_val

#     if isinstance(shifted_val, Measurement):
#         return f"${shifted_val:.0e}$" if force_exponent else f"${shifted_val}$"
#         s = shifted_val if not force_exponent else shifted_val.to_str_bracket(0)
#         return f"${s}$"  # Math mode for Measurement strings

#     return f"${np.round(shifted_val, 10)}$"

# def format_column_data(column_data, metadata: ColumnMetadata) -> List[CellValue]:
#     force_exp: bool = metadata.force_exponent or False;
#     return [format_value(val, metadata.display_exponent, force_exp) for val in column_data]

# def format_latex_string(value: str) -> str:
#     """Apply LaTeX-specific replacements to a string."""
#     latex_patterns = {
#         r'\bphi\b': r'$\varphi$',
#         r"_{?(\w+)}?": r"_{\1}",
#         r"\^{?(\w+)}?": r"^{\1}",
#     }
#     for pattern, replacement in latex_patterns.items():
#         value = re.sub(pattern, replacement, value)
#     return value
