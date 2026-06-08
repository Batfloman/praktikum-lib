from typing import Any, Sequence, overload

import numpy as np
import pandas as pd

from ..structs.dataCluster import DataCluster
from .plot_state import Plot
from .types import ScatterResult, SupportedValues


@overload
def scatter(
    x: Sequence[SupportedValues] | np.ndarray,
    y: Sequence[SupportedValues] | np.ndarray,
    /,
    *,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...


@overload
def scatter(
    *,
    x: Sequence[SupportedValues] | np.ndarray,
    y: Sequence[SupportedValues] | np.ndarray,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...


@overload
def scatter(
    data: DataCluster,
    x_index: str,
    y_index: str,
    /,
    *,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...


@overload
def scatter(
    data: pd.DataFrame,
    x_index: str,
    y_index: str,
    /,
    *,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...


@overload
def scatter(
    data: DataCluster,
    *,
    x: str,
    y: str,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...


@overload
def scatter(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult: ...
