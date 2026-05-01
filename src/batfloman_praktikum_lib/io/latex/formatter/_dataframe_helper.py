from typing import Optional

import pandas as pd

from batfloman_praktikum_lib.io.table_metadata import (
    ALIGNMENT_VALUES,
    DEFAULT_ALIGNMENT,
    TableColumnMetadata,
    TableMetadataManager,
)

from ..optionTypes import TableOptions


def normalize_metadata_manager(
    metadata: Optional[TableMetadataManager | dict[str, TableColumnMetadata]],
) -> TableMetadataManager:
    if metadata is None:
        return TableMetadataManager()

    if isinstance(metadata, TableMetadataManager):
        return metadata

    manager = TableMetadataManager()
    for index, md in metadata.items():
        manager.update_metadata(index, md)
    return manager


def resolve_indices(df: pd.DataFrame, options: TableOptions) -> list[str]:
    use_indices = options.get("use_indices")
    exclude_indices = options.get("exclude_indices")

    if use_indices is not None:
        missing = set(use_indices) - set(df.columns)
        if missing:
            raise ValueError(f"{missing} not in {list(df.columns)}")

    if exclude_indices is not None:
        missing = set(exclude_indices) - set(df.columns)
        if missing:
            raise ValueError(f"{missing} not in {list(df.columns)}")

    indices = list(df.columns) if use_indices is None else list(use_indices)
    excluded = set(exclude_indices or [])
    return [index for index in indices if index not in excluded]
def _get_single_column_format(
    index: str,
    metadata: Optional[TableColumnMetadata],
) -> str:
    if not metadata:
        return DEFAULT_ALIGNMENT

    alignment = getattr(metadata, "alignment", DEFAULT_ALIGNMENT)
    if alignment not in ALIGNMENT_VALUES:
        raise ValueError(
            f"Invalid alignment '{alignment}' for index `{index}`. Must be one of {ALIGNMENT_VALUES}."
        )

    s = ""
    if getattr(metadata, "left_border", False):
        s += "|"
    s += alignment
    if getattr(metadata, "right_border", False):
        s += "|"

    return s


def get_column_format(
    indices: list[str],
    metadata_manager: Optional[TableMetadataManager] = None,
) -> str:
    if not metadata_manager:
        return DEFAULT_ALIGNMENT * len(indices)

    individual_formats = [
        _get_single_column_format(index, metadata_manager.get_metadata(index))
        for index in indices
    ]
    return "".join(individual_formats)
