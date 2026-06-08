import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ..structs.measurementBase import MeasurementBase
from .types import SupportedValues


def _normalize_column_name(name: object) -> str:
    return str(name).replace(" ", "")


def dataframe_column(data: pd.DataFrame, index: str) -> pd.Series:
    normalized_columns = {
        _normalize_column_name(column): column
        for column in data.columns
    }
    normalized_index = _normalize_column_name(index)

    if normalized_index not in normalized_columns:
        available = ", ".join(repr(column) for column in data.columns)
        raise ValueError(f"Column {index!r} not found in DataFrame. Available columns: {available}.")

    return data[normalized_columns[normalized_index]]


def extract_value_error(
    values: Sequence[SupportedValues],
) -> tuple[np.ndarray, np.ndarray]:
    resolved_values = []
    errors = []

    for item in values:
        if isinstance(item, np.str_):
            try:
                item = float(item) if "." in item else int(item)
            except Exception:
                raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")

        if isinstance(item, (float, int, np.integer)):
            resolved_values.append(item)
            errors.append(0)
        elif isinstance(item, MeasurementBase):
            resolved_values.append(item.value)
            errors.append(item.error)
        else:
            raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")

    return np.array(resolved_values), np.array(errors)


def filter_nan_values(
    x: Sequence[SupportedValues] | np.ndarray,
    y: Sequence[SupportedValues] | np.ndarray,
    warn_filter_nan: bool = True,
):
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_clean = x[mask]
    y_clean = y[mask]

    if warn_filter_nan and len(x) != len(x_clean):
        warnings.warn("\nFiltered NaN values.", RuntimeWarning)

    return x_clean, y_clean
