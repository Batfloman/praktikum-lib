from typing import Optional, Any, Iterable
import numbers
import pandas as pd

from ..table_metadata import TableColumnMetadata, TableMetadataManager
from .save_impl import save_numbers, save_Measurement, save_pd_dataframe, save_DataCluster

def save_latex(
    obj: Any,
    path: str,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
    # tables
    tableMetadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] = None,
    use_indices: Optional[list[str]] = None,
    exclude_indices: Optional[Iterable[str]] = None,
    # values
    format_spec: str = "",
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    with_error: bool = True,
) -> str:
    from batfloman_praktikum_lib.structs.dataCluster import DataCluster
    from batfloman_praktikum_lib.structs.measurement import Measurement

    if isinstance(obj, numbers.Real):
        return save_numbers(obj, path, 
            print_success_msg= print_success_msg,
            auto_create_dirs= auto_create_dirs,
            # values
            format_spec=format_spec,
            unit = unit,
            use_si_prefix = use_si_prefix,
            fixed_exponent = fixed_exponent,
        )
    if isinstance(obj, Measurement):
        return save_Measurement(obj, path,
            print_success_msg= print_success_msg,
            auto_create_dirs= auto_create_dirs,
            # values
            format_spec=format_spec,
            unit = unit,
            use_si_prefix = use_si_prefix,
            fixed_exponent = fixed_exponent,
            with_error = with_error,
        )
    if isinstance(obj, pd.DataFrame):
        return save_pd_dataframe(obj, path,
            print_success_msg= print_success_msg,
            auto_create_dirs= auto_create_dirs,
            # values
            tableMetadata= tableMetadata,
            use_indices = use_indices,
            exclude_indices = exclude_indices,
        )
    if isinstance(obj, DataCluster):
        return save_DataCluster(obj, path,
            print_success_msg= print_success_msg,
            auto_create_dirs= auto_create_dirs,
            # values
            tableMetadata= tableMetadata,
            use_indices = use_indices,
            exclude_indices = exclude_indices,
        )

    raise NotImplementedError(f"LaTeX saving for type {type(obj).__name__} not implemented!")

