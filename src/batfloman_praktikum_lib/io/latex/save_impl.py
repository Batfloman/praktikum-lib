from typing import Iterable, Optional
import numbers
import pandas as pd

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement

from ..table_metadata import TableColumnMetadata, TableMetadataManager
from ..validation import create_dirs, ensure_extension
from .formatter import format_number_latex_str, format_table_header, format_table_value, get_column_format

def save_numbers(
    value: numbers.Real,
    path: str, 
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
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

def save_Measurement(
    value: Measurement,
    path: str, 
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
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

def save_pd_dataframe(
    df: pd.DataFrame,
    path: str, 
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] = None,
    use_indices: Optional[list[str]] = None,
    exclude_indices: Optional[Iterable[str]] = None,
) -> str:
    tableMetadata = tableMetadata or TableMetadataManager()
    exclude_indices = exclude_indices or []

    if isinstance(tableMetadata, dict):
        md = TableMetadataManager()
        for k, v in tableMetadata.items():
            md.update_metadata(k, v)
        tableMetadata = md

    indices = list(df.columns) if use_indices is None else use_indices
    indices = [c for c in indices if c not in exclude_indices]

    column_format = get_column_format(indices, tableMetadata)
    formatted_headers = [
        format_table_header(col, tableMetadata.get_metadata(col)) for col in indices
    ]

    latex_str = ""
    latex_str += r"\begin{tabular}{" + column_format + "}\n"

    # header
    latex_str += "\t" + r"\toprule" + "\n"
    latex_str += "\t" + (" & ".join(formatted_headers)) + r"\\" + "\n"
    latex_str += "\t" + r"\midrule" + "\n"

    #values
    for _, row in df.iterrows():
        processed_row = [
            format_table_value(v, tableMetadata.get_metadata(k)) 
            for k, v in zip(indices, row[indices])
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

def save_DataCluster(
    obj: DataCluster,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] = None,
    use_indices: Optional[list[str]] = None,
    exclude_indices: Optional[Iterable[str]] = None,
) -> str:
    latex_str = save_pd_dataframe(obj.to_dataframe(), path,
        print_success_msg = False,
        auto_create_dirs = auto_create_dirs,
        # tables
        tableMetadata = tableMetadata,
        use_indices=use_indices,
        exclude_indices=exclude_indices,
    )

    if print_success_msg:
        print(f"Succesfully saved DataCluster to {path}")

    return latex_str
