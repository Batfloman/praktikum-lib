from collections.abc import Callable
from typing import Any, Literal, Sequence, overload

import pandas as pd

from ..structs.dataCluster import DataCluster
from ..graph_fit.fitResult import FitResult
from .plot_state import Plot
from .types import PlotResult, SupportedValues

type ErrorBandMode = bool | Literal["auto"]

def fit_quality_label(
    fit_result: FitResult,
    *,
    decimals: int = 4,
    decimal_comma: bool = False,
) -> str: ...


@overload
def plot(
    x: Sequence[SupportedValues],
    y: Sequence[SupportedValues],
    /,
    *,
    plot: Plot | None = None,
    change_viewport: bool = True,
    with_error: bool = True,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    fit_result: FitResult,
    /,
    *,
    plot: Plot | None = None,
    interval: tuple[float, float] | None = None,
    change_viewport: bool = True,
    with_error: ErrorBandMode = "auto",
    log_scale: bool = False,
    min_error_band_fraction: float = 0.05,
    show_fit_quality_label: bool = True,
    fit_quality_label_decimal_comma: bool = True,
    fit_quality_label_decimals: int = 4,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    func: Callable[..., Any],
    /,
    *,
    plot: Plot | None = None,
    interval: tuple[float, float] | None = None,
    change_viewport: bool = True,
    with_error: ErrorBandMode = "auto",
    log_scale: bool = False,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    data: DataCluster,
    x_index: str,
    y_index: str,
    /,
    *,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    data: pd.DataFrame,
    x_index: str,
    y_index: str,
    /,
    *,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    data: DataCluster,
    *,
    x: str,
    y: str,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult: ...


@overload
def plot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult: ...
