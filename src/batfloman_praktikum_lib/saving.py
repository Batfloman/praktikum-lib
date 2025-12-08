from typing import Union, Optional, Literal
import re
import numpy as np
from numpy import number

from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.tables.latex_table.formatter import _format_unit as format_unit
from batfloman_praktikum_lib.path_managment import create_dirs, ensure_extension
from batfloman_praktikum_lib.significant_rounding.formatter import UncertaintyNotation

def _extract_precision(format_spec: str) -> int | None:
    """
    Extract the precision N from a format spec like '.2f', '12.3e', '>.4g', etc.
    Returns None if no precision is specified.
    """
    m = re.search(r'\.(\d+)', format_spec)
    if m:
        return int(m.group(1))
    return None

def export_value_as_latex(
    value: number,
    path: str,
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    print_success_msg: bool = True,
    format_spec: str = "",
):
      # 🔹 Handle custom e3 format
    if format_spec.endswith("e3"):
        # Determine the nearest exponent multiple of 3
        exp = int(np.floor(np.log10(abs(value)) / 3) * 3) if value != 0 else 0
        prec = _extract_precision(format_spec) or 0
        num_text = f"{value / 10**exp:.{prec}f}"  # strip 'e3' for precision
        unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
    else:
        formatted = format(value, format_spec)
        if "e" in formatted:
            num_text, exp_str = formatted.split("e")
            exp = int(exp_str)
            unit_text = format_unit(unit, exponent=exp, use_si_prefix=use_si_prefix)
        else:
            num_text = formatted
            unit_text = format_unit(unit, use_si_prefix=use_si_prefix)

    if unit_text != "":
        unit_text = fr"\,{unit_text}"

    latex_str = fr"\ensuremath{{{num_text}}}{unit_text}"

    # 🔹 Write to file
    path = ensure_extension(path, ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"Succesfully saved `{value}` to {path} as [{latex_str}]")

    return latex_str

    # if unit_text is not "":
    #     unit_text = fr"\,{unit_text}"
    #
    # latex_str = fr"\ensuremath{{{num_text}}}{unit_text}"
    #
    # # 🔹 Write to file
    # path = ensure_extension(path, ".tex")
    # with open(path, "w", encoding="utf-8") as f:
    #     f.write(latex_str)
    #
    # if print_success_msg:
    #     print(f"Succesfully saved `{formatted}` to {path} as [{latex_str}]")
    #
    # return latex_str

def save_latex(
    obj: Union[float, number, Measurement],
    filepath: str,

    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    print_success_msg: bool = True,
    format_spec: str = "",
    with_error: bool = True,
    mode: Union[UncertaintyNotation, Literal["pm", "brk"]] = UncertaintyNotation.Parentheses,
):
    filepath = ensure_extension(filepath, ".tex")
    create_dirs(filepath)

    if isinstance(obj, (number, float)):
        export_value_as_latex(
            obj, 
            path = filepath,
            unit=unit, 
            use_si_prefix=use_si_prefix, 
            print_success_msg=print_success_msg, 
            format_spec=format_spec
        )       
    elif isinstance(obj, Measurement):
        obj.save_latex(
            path=filepath,
            unit = unit,
            use_si_prefix = use_si_prefix,
            print_success_msg = print_success_msg,
            mode = mode,
            format_spec = format_spec,
            with_error = with_error,
        )
    else:
        raise NotImplementedError(f"LaTeX saving for type {type(obj)} not Implemented!")
