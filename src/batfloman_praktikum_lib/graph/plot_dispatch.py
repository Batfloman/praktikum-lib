from collections.abc import Callable
from functools import singledispatch
from typing import Any, Literal, Sequence, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..graph_fit.fitResult import FitResult, format_fit_quality
from ..structs.dataCluster import DataCluster
from .helpers import dataframe_column, extract_value_error, filter_nan_values
from .plot_state import Plot, resolve_plot
from .types import PlotResult, SupportedValues

type ErrorBandMode = bool | Literal["auto"]


def fit_quality_label(
    fit_result: FitResult,
    *,
    decimals: int = 4,
    decimal_comma: bool = False,
) -> str:
    return format_fit_quality(
        fit_result,
        decimals=decimals,
        decimal_comma=decimal_comma,
        latex=True,
    )


def _auto_show_fit_error_band(
    y_smooth: Sequence[SupportedValues],
    ax: Axes,
    *,
    min_error_band_fraction: float = 0.005,
) -> bool:
    y_values, y_err = extract_value_error(y_smooth)
    band_width = 2 * np.asarray(y_err, dtype=float)

    if not np.any(band_width > 0):
        return False

    y_values = np.asarray(y_values, dtype=float)
    finite_y = y_values[np.isfinite(y_values)]
    if finite_y.size == 0:
        return False

    line_span = np.nanmax(finite_y) - np.nanmin(finite_y)
    axis_ymin, axis_ymax = ax.get_ylim()
    axis_span = axis_ymax - axis_ymin if ax.has_data() else np.nan
    y_span = axis_span if np.isfinite(axis_span) and axis_span > 0 else line_span

    if not np.isfinite(y_span) or y_span <= 0:
        return True

    return np.nanpercentile(band_width, 95) >= min_error_band_fraction * y_span


def _plot_xy(
    x: Sequence[SupportedValues],
    y: Sequence[SupportedValues],
    plot: Plot,
    change_viewport: bool = True,
    with_error: bool = True,
    **kwargs: Any,
) -> PlotResult:
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")

    _, ax = plot

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_values, _ = extract_value_error(x)
    y_values, y_err = extract_value_error(y)

    zorder = kwargs.pop("zorder", 3)
    line, = ax.plot(x_values, y_values, zorder=zorder, **kwargs)

    fill = None
    if with_error and np.any(y_err > 0):
        y_min = y_values - y_err
        y_max = y_values + y_err
        fill = ax.fill_between(
            x_values,
            y_min,
            y_max,
            color=line.get_color(),
            alpha=0.3,
            label=fr"{line.get_label()} $\pm 1 \sigma$" if line.get_label() else None,
            zorder=zorder - 0.1,
        )

    if not change_viewport:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return PlotResult(line=line, fill=fill, plot=plot)


def plot(data: object, *args: object, **kwargs: Any) -> PlotResult:
    return _plot_dispatch(data, *args, **kwargs)


@singledispatch
def _plot_dispatch(data: object, *args: object, **kwargs: Any) -> PlotResult:
    if callable(data) and not args:
        target_plot = resolve_plot(kwargs.pop("plot", None))
        return _plot_callable(data, plot=target_plot, **kwargs)

    if len(args) != 1:
        raise TypeError(
            "graph.plot() expects x/y values, a callable, a FitResult, or a DataCluster."
        )

    target_plot = resolve_plot(kwargs.pop("plot", None))
    return _plot_xy(
        cast(Sequence[SupportedValues], data),
        cast(Sequence[SupportedValues], args[0]),
        plot=target_plot,
        **kwargs,
    )


@_plot_dispatch.register
def _plot_fit_result(
    fit_result: FitResult,
    *args: object,
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
) -> PlotResult:
    if args:
        raise TypeError("graph.plot(FitResult) does not accept positional y data.")

    target_plot = resolve_plot(plot)
    _, ax = target_plot
    if min_error_band_fraction < 0:
        raise ValueError("min_error_band_fraction must be non-negative.")
    if with_error != "auto" and not isinstance(with_error, bool):
        raise ValueError('with_error must be True, False, or "auto".')

    xmin, xmax = ax.get_xlim() if interval is None else interval
    if log_scale:
        xmin = np.log10(max(xmin, 1e-10))
        xmax = np.log10(xmax)

    x_smooth = np.linspace(xmin, xmax, 10000)
    if log_scale:
        x_smooth = 10**x_smooth
    x_smooth = list(x_smooth)

    y_smooth = [fit_result.func(x) for x in x_smooth]

    if with_error == "auto":
        with_error = _auto_show_fit_error_band(
            y_smooth,
            ax,
            min_error_band_fraction=min_error_band_fraction,
        )

    if show_fit_quality_label and "label" not in kwargs:
        kwargs["label"] = fit_quality_label(
            fit_result,
            decimals=fit_quality_label_decimals,
            decimal_comma=fit_quality_label_decimal_comma,
        )

    return _plot_xy(
        x_smooth,
        y_smooth,
        plot=target_plot,
        change_viewport=change_viewport,
        with_error=with_error,
        **kwargs,
    )


def _plot_callable(
    func: Callable[..., Any],
    *,
    plot: Plot,
    interval: tuple[float, float] | None = None,
    change_viewport: bool = True,
    with_error: ErrorBandMode = "auto",
    log_scale: bool = False,
    **kwargs: Any,
) -> PlotResult:
    _, ax = plot
    if with_error != "auto" and not isinstance(with_error, bool):
        raise ValueError('with_error must be True, False, or "auto".')

    xmin, xmax = ax.get_xlim() if interval is None else interval
    if log_scale:
        xmin = np.log10(max(xmin, 1e-10))
        xmax = np.log10(xmax)

    x_smooth = np.linspace(xmin, xmax, 10000)
    if log_scale:
        x_smooth = 10**x_smooth
    x_smooth = list(x_smooth)

    y_smooth = [func(x) for x in x_smooth]

    if with_error == "auto":
        with_error = True

    return _plot_xy(
        x_smooth,
        y_smooth,
        plot=plot,
        change_viewport=change_viewport,
        with_error=with_error,
        **kwargs,
    )


@_plot_dispatch.register
def _plot_data_cluster(
    data: DataCluster,
    *args: object,
    x: str | None = None,
    y: str | None = None,
    x_index: str | None = None,
    y_index: str | None = None,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult:
    x_index = x_index
    y_index = y_index

    if len(args) == 2:
        if x is not None or y is not None or x_index is not None or y_index is not None:
            raise TypeError("Use either positional x/y indices or x_index/y_index, not both.")
        x_index, y_index = str(args[0]), str(args[1])
    elif len(args) == 0:
        x_index = x if x_index is None else x_index
        y_index = y if y_index is None else y_index
    else:
        raise TypeError("graph.plot(DataCluster) expects x and y column indices.")

    if x_index is None or y_index is None:
        raise TypeError("graph.plot(DataCluster) requires x and y column indices.")

    target_plot = resolve_plot(plot)
    if str(x_index) not in data.get_column_names():
        raise ValueError(f'Column "{x_index}" not found in data.')
    if str(y_index) not in data.get_column_names():
        raise ValueError(f'Column "{y_index}" not found in data.')

    x_values = data.column(str(x_index))
    y_values = data.column(str(y_index))
    x_values, y_values = filter_nan_values(
        x_values,
        y_values,
        warn_filter_nan=warn_filter_nan,
    )

    return _plot_xy(
        x=x_values,
        y=y_values,
        plot=target_plot,
        with_error=with_error,
        change_viewport=change_viewport,
        **kwargs,
    )


@_plot_dispatch.register
def _plot_dataframe(
    data: pd.DataFrame,
    *args: object,
    x: str | None = None,
    y: str | None = None,
    x_index: str | None = None,
    y_index: str | None = None,
    plot: Plot | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> PlotResult:
    if len(args) == 2:
        if x is not None or y is not None or x_index is not None or y_index is not None:
            raise TypeError("Use either positional x/y indices or x_index/y_index, not both.")
        x_index, y_index = str(args[0]), str(args[1])
    elif len(args) == 0:
        x_index = x if x_index is None else x_index
        y_index = y if y_index is None else y_index
    else:
        raise TypeError("graph.plot(DataFrame) expects x and y column indices.")

    if x_index is None or y_index is None:
        raise TypeError("graph.plot(DataFrame) requires x and y column indices.")

    target_plot = resolve_plot(plot)
    x_values = dataframe_column(data, str(x_index)).to_numpy(dtype=object)
    y_values = dataframe_column(data, str(y_index)).to_numpy(dtype=object)
    x_values, y_values = filter_nan_values(
        x_values,
        y_values,
        warn_filter_nan=warn_filter_nan,
    )

    return _plot_xy(
        x=x_values,
        y=y_values,
        plot=target_plot,
        with_error=with_error,
        change_viewport=change_viewport,
        **kwargs,
    )


__all__ = [
    "ErrorBandMode",
    "fit_quality_label",
    "plot",
]
