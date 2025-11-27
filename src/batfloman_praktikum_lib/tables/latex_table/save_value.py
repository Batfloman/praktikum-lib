from typing import Optional

from .formatter import _format_unit as format_unit
from ..validation import ensure_extension

def export_value_as_latex(
    value: float,
    path: str,
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    print_success_msg: bool = True,
    format_spec: str = "",
):
    formatted = format(value, format_spec); 

    # 🔹 Handle scientific notation (e.g. 1.23e-3)
    if "e" in formatted:
        num_text, exp_str = formatted.split("e")
        exp = int(exp_str)
        unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
    else:
        num_text = formatted
        unit_text = format_unit(unit, use_si_prefix=use_si_prefix)

    if unit_text is not "":
        unit_text = fr"\,{unit_text}"

    latex_str = fr"\ensuremath{{{num_text}}}{unit_text}"

    # 🔹 Write to file
    path = ensure_extension(path, ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"Succesfully saved `{formatted}` to {path} as [{latex_str}]")

    return latex_str
