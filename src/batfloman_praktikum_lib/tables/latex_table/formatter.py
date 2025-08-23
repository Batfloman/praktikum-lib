from ..metadata import ColumnMetadata
from ...structs.measurement import Measurement

from typing import List, Union, Optional
import numpy as np
import re

# typing alias
CellValue = Union[float, str, Measurement]

def format_symbol(name: str) -> str:
    # crude heuristic: if it contains `_` or `^` or is a single letter → math mode
    pattern = r"^[A-Za-z](?:([_^](?:\{[^{}]+\}|[A-Za-z0-9]+)))*$"
    if re.match(pattern, name):
        return f"${name}$"
    return name

def format_unit(unit: str) -> str:
    """
    Format units for LaTeX without using full math mode.
    Example: 'm/s^2' -> 'm/s\textsuperscript{2}'
    """
    # Hochzahlen automatisch in \textsuperscript umwandeln
    unit = re.sub(r"\^(\d+)", r"\\textsuperscript{\1}", unit) 
    return unit

def format_header(metadata: ColumnMetadata, index: str) -> str:
    """Returns the formatted header string."""
    name = metadata.name or index
    name = format_symbol(name)

    unit = metadata.unit
    exponent = metadata.display_exponent

    if exponent is None or exponent == 0:
        exponent_text = ""
    else:
        exponent_text = fr"$10^{{{exponent}}}$ " # spacing for [exponent unit] text

    if unit is None or unit == "":
        unit_text = f"[{exponent_text}]" if exponent_text else ""
    else:
        unit_text = f"[{exponent_text}{format_unit(unit)}]" # uses space from above

    return f"{name} {unit_text}".strip()

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
