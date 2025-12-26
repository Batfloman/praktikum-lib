from functools import singledispatch
from typing import Optional, Any
import numbers
import pandas as pd

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement
from ..validation import ensure_extension
from ..table_metadata import TableColumnMetadata
from .formatter import format_number_latex_str

@singledispatch
def save_latex(
    obj: Any,
    path: str,
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    raise NotImplementedError(f"LaTeX saving for type {type(obj)} not Implemented!")

@save_latex.register
def _(
    value: numbers.Real,
    path: str, 
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    latex_str = format_number_latex_str(value, 
        unit=unit,
        use_si_prefix=use_si_prefix,
        fixed_exponent=fixed_exponent,
        format_spec=format_spec
    )

    # Write to file
    path = ensure_extension(path, ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"Succesfully saved `{value}` to {path} as [{latex_str}]")

    return latex_str

@save_latex.register
def _(
    value: Measurement,
    path: str, 
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    val = value if with_error else value.value

    latex_str = format_number_latex_str(val, 
        unit=unit,
        use_si_prefix=use_si_prefix,
        fixed_exponent=fixed_exponent,
        format_spec=format_spec
    )

    # Write to file
    path = ensure_extension(path, ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"Succesfully saved `{value}` to {path} as [{latex_str}]")

    return latex_str

# ==================================================

@save_latex.register
def _(
    obj: pd.DataFrame,
    path: str, 
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    # Build formatted headers if metadata is provided
    if tableMetadata:
        formatted_headers = [
            format_header(col, tableMetadata.get_metadata(col)) for col in df.columns
        ]
    else:
        formatted_headers = header  # can be True/False



    # write to file
    path = ensure_extension(path, ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    # done
    if print_success_msg:
        print(f"Succesfully saved `{repr(obj)}` to {path}")

    return latex_str

# ==================================================

@save_latex.register
def _(
    obj: DataCluster,
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    return obj

    if print_success_msg:
        print(f"Succesfully saved `{value}` to {path} as [{latex_str}]")

    return latex_str

