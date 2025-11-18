import pandas as pd

from typing import Any, Callable, List, Optional, Tuple, Union, NamedTuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection

from .plotNScatter import scatter, ScatterResult

def _normalize_column_name(name):
    return name.replace(" ", "")

def scatter_dataframe(
    data: pd.DataFrame,
    x_index: str, 
    y_index: str,
    with_error: bool = True,
    plot: Optional[Tuple[Figure, Any]] = None,
    **kwargs
) -> ScatterResult:
    normalized_columns = {_normalize_column_name(col): col for col in data.columns}

    if _normalize_column_name(x_index) not in normalized_columns:
        raise ValueError(f"Column \"{x_index}\" not found in DataFrame.")
    if _normalize_column_name(y_index) not in normalized_columns:
        raise ValueError(f"Column \"{y_index}\" not found in DataFrame.")

    x = data[normalized_columns[_normalize_column_name(x_index)]]
    y = data[normalized_columns[_normalize_column_name(y_index)]]

    return scatter(x, y, with_error, plot=plot, **kwargs)
