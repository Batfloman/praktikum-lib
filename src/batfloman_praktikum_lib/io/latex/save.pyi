from typing import overload, Optional, Iterable
import pandas as pd
import numbers

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from ..table_metadata import TableColumnMetadata, TableMetadataManager

@overload
def save_latex(
    obj: numbers.Real,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
) -> str: ...

@overload
def save_latex(
    obj: MeasurementBase,
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
) -> str: ...

@overload
def save_latex(
    obj: pd.DataFrame,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] = None,
    use_indices: Optional[list[str]] = None,
    exclude_indices: Optional[Iterable[str]] = None,
) -> str: ...

@overload
def save_latex(
    obj: DataCluster,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] = None,
    use_indices: Optional[list[str]] = None,
    exclude_indices: Optional[Iterable[str]] = None,
) -> str: ...
