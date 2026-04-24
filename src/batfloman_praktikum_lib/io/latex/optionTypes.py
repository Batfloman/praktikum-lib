from typing import Optional, Sequence, TypedDict

from batfloman_praktikum_lib.io.termColors import bcolors
from batfloman_praktikum_lib.io.table_metadata import TableColumnMetadata, TableMetadataManager


class ValueOptions(TypedDict, total=False):
    format_spec: str
    unit: Optional[str]
    use_si_prefix: bool
    fixed_exponent: Optional[int]
    with_error: bool


class TableOptions(TypedDict, total=False):
    metadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]] 
    use_indices: Optional[Sequence[str]]
    exclude_indices: Optional[Sequence[str]]
    unit_separator: str


DEFAULT_VALUE_OPTIONS: ValueOptions = {
    "format_spec": "",
    "unit": None,
    "use_si_prefix": True,
    "fixed_exponent": None,
    "with_error": True,
}

DEFAULT_TABLE_OPTIONS: TableOptions = {
    "metadata": None,
    "use_indices": None,
    "exclude_indices": None,
    "unit_separator": " in ",
}

VALUE_OPTION_KEYS = set(DEFAULT_VALUE_OPTIONS.keys())
TABLE_OPTION_KEYS = set(DEFAULT_TABLE_OPTIONS.keys())


def _warn_unknown_keys(options: dict, valid_keys: set[str], namespace: str) -> None:
    unknown_keys = sorted(set(options.keys()) - valid_keys)
    if not unknown_keys:
        return

    valid_keys_str = ", ".join(sorted(valid_keys))
    unknown_keys_str = ", ".join(unknown_keys)
    print(
        f"{bcolors.WARNING}Warning{bcolors.ENDC}: "
        f"Ignoring unknown {namespace} option keys: {unknown_keys_str}. "
        f"{bcolors.DIM}Valid keys: {valid_keys_str}{bcolors.ENDC}"
    )


def normalize_value_options(options: ValueOptions | None = None) -> ValueOptions:
    _warn_unknown_keys(options or {}, VALUE_OPTION_KEYS, "value")
    return {
        **DEFAULT_VALUE_OPTIONS,
        **(options or {}),
    }


def normalize_table_options(options: TableOptions | None = None) -> TableOptions:
    _warn_unknown_keys(options or {}, TABLE_OPTION_KEYS, "table")
    return {
        **DEFAULT_TABLE_OPTIONS,
        **(options or {}),
    }
