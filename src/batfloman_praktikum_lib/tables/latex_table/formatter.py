from ..metadata import ColumnMetadata
from ...structs.measurement import Measurement

from typing import List, Union, Optional
import numpy as np
import re

# typing alias
CellValue = Union[float, str, Measurement]

def format_symbol(name: str) -> str:
    # crude heuristic: if it contains `_` or `^` or is a single letter → math mode
    pattern = r"^[A-Za-z](?:([_^](?:\{[^{}]+\}|[A-Za-z0-9]+)))*"
    if re.match(pattern, name):
        return f"${name}$"
    return name

# SI_PREFIXES = {
#      9: "G",  # giga
#      6: "M",  # mega
#      3: "k",  # kilo
#     -3: "m",  # milli
#     -6: r"\textmu ",  # micro (space important so that latex \textmuUnit does not result in an unknown command!)
#     -9: "n",  # nano
#     -12: "p", # pico
# }
SI_PREFIXES = {
     9: r"\giga",  # giga
     6: r"\mega",  # mega
     3: r"\kilo",  # kilo
    -3: r"\milli",  # milli
    -6: r"\micro ",  # micro (space important so that latex \textmuUnit does not result in an unknown command!)
    -9: r"\nano",  # nano
    -12: r"\pico", # pico
}

def format_exponent(exponent: int | None, use_si_prefix: bool = True) -> str:
    if exponent in (None, 0):
        return "";
    
    return SI_PREFIXES[exponent] if (use_si_prefix and exponent in SI_PREFIXES) else fr"\ensuremath{{ 10^{{{exponent}}} }}";

def format_unit(unit: str | None, exponent: Optional[int] = None, use_si_prefix: bool = True) -> str:
    """
    Returns a string suitable for siunitx: \si{unit}.
    """
    if unit is None or unit == "":
        return ""

    if not exponent:
        return fr"\si{{ {unit} }}"
    
    if use_si_prefix and exponent in SI_PREFIXES:
        prefix = SI_PREFIXES[exponent]
        return fr"\si {{ {prefix} {unit} }}"
    else:
        return fr"\ensuremath{{ 10^{{{exponent}}} }}\,\si {{ {unit} }}"

def format_header(metadata: ColumnMetadata, index: str) -> str:
    """Returns the formatted header string."""
    name = metadata.name or format_symbol(index)

    unit = metadata.unit
    exponent = metadata.display_exponent
    use_si = True if (metadata.use_si_prefix is None) else metadata.use_si_prefix

    unit_text = format_unit(unit, exponent=exponent, use_si_prefix=use_si)
    if unit_text != "":
        unit_text = f" in {unit_text}"

    return rf"{name}{unit_text}"

def format_value(
    val: Union[float, str, Measurement],
    display_exponent: Optional[int] = None,
    force_exponent: bool = False
) -> Union[float, str, Measurement]:
    if str(val).strip() == "-":
        return "-"
    num_val = float(val) if isinstance(val, str) else val
    shifted_val = num_val / 10**display_exponent if display_exponent else num_val

    if isinstance(shifted_val, Measurement):
        s = shifted_val if not force_exponent else shifted_val.to_str_bracket(0)
        return f"${s}$"  # Math mode for Measurement strings

    return f"${np.round(shifted_val, 10)}$"

def format_column_data(column_data, metadata: ColumnMetadata) -> List[CellValue]:
    force_exp: bool = metadata.force_exponent or False;
    return [format_value(val, metadata.display_exponent, force_exp) for val in column_data]

def format_latex_string(value: str) -> str:
    """Apply LaTeX-specific replacements to a string."""
    latex_patterns = {
        r'\bphi\b': r'$\varphi$',
        r"_{?(\w+)}?": r"_{\1}",
        r"\^{?(\w+)}?": r"^{\1}",
    }
    for pattern, replacement in latex_patterns.items():
        value = re.sub(pattern, replacement, value)
    return value
