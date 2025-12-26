from functools import singledispatch
from typing import Optional, Any
import numbers
import pandas as pd

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement
from ..validation import create_dirs, ensure_extension
from ..table_metadata import TableColumnMetadata, TableMetadataManager
from .formatter import format_number_latex_str, format_table_header, format_table_value, get_column_format


@singledispatch
def save_latex(
    obj: Any,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: TableMetadataManager | dict[str, TableColumnMetadata] = TableMetadataManager(),
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    raise NotImplementedError(f"LaTeX saving for type {type(obj)} not Implemented!")

# ==================================================

@save_latex.register
def _(
    value: numbers.Real,
    path: str, 
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: TableMetadataManager | dict[str, TableColumnMetadata] = TableMetadataManager(),
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
    if auto_create_dirs:
        create_dirs(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"Succesfully saved `{value}` to {path} as [{latex_str}]")

    return latex_str

# ==================================================

@save_latex.register
def _(
    value: Measurement,
    path: str, 
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: TableMetadataManager | dict[str, TableColumnMetadata] = TableMetadataManager(),
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
    if auto_create_dirs:
        create_dirs(path)
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
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: TableMetadataManager | dict[str, TableColumnMetadata] = TableMetadataManager(),
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    if isinstance(tableMetadata, dict):
        md = TableMetadataManager()
        for k, v in tableMetadata.items():
            md.update_metadata(k, v)
        tableMetadata = md

    column_format = get_column_format(obj.columns, tableMetadata)
    formatted_headers = [
        format_table_header(col, tableMetadata.get_metadata(col)) for col in obj.columns
    ]

    latex_str = ""
    latex_str += r"\begin{tabular}{" + column_format + "}\n"

    # header
    latex_str += "\t" + r"\toprule" + "\n"
    latex_str += "\t" + (" & ".join(formatted_headers)) + r"\\" + "\n"
    latex_str += "\t" + r"\midrule" + "\n"

    #values
    for _, row in obj.iterrows():
        processed_row = [
            format_table_value(v, tableMetadata.get_metadata(k)) for k,v in zip(obj.columns, row)
        ]
        latex_str += "\t" + (" & ".join(processed_row)) + r"\\" + "\n"
    latex_str += "\t" + r"\bottomrule" + "\n"

    latex_str += r"\end{tabular}" + "\n"

    # write to file
    path = ensure_extension(path, ".tex")
    if auto_create_dirs:
        create_dirs(path)
    with open(path, 'w') as f:
        f.write(latex_str)

    # done
    if print_success_msg:
        print(f"Succesfully saved table to {path}")

    return latex_str

# ==================================================

@save_latex.register
def _(
    obj: DataCluster,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: TableMetadataManager | dict[str, TableColumnMetadata] = TableMetadataManager(),
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    return save_latex(
        obj.to_dataframe(),
        path,
        # named
        print_success_msg = print_success_msg,
        auto_create_dirs = auto_create_dirs,
        # tables
        tableMetadata = tableMetadata
        # rest is not used
    )
