from functools import singledispatch
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from ..structs.dataCluster import DataCluster
from .helpers import dataframe_column, extract_value_error
from .plot_state import Plot, resolve_plot
from .types import ScatterResult, SupportedValues


def _scatter_xy(
    x: Sequence[SupportedValues] | np.ndarray,
    y: Sequence[SupportedValues] | np.ndarray,
    plot: Plot,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult:
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")

    x_raw = np.asarray(x, dtype=object)
    y_raw = np.asarray(y, dtype=object)

    if x_interval is not None:
        xmin, xmax = x_interval
        if xmin > xmax:
            raise ValueError("x_interval must satisfy x_interval[0] <= x_interval[1].")

        x_interval_values, _ = extract_value_error(x_raw)
        interval_mask = (x_interval_values >= xmin) & (x_interval_values <= xmax)
        x_raw = x_raw[interval_mask]
        y_raw = y_raw[interval_mask]

    _, ax = plot

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_values, x_err = extract_value_error(x_raw)
    y_values, y_err = extract_value_error(y_raw)

    nan_mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    if warn_filter_nan and len(x_values) != np.count_nonzero(nan_mask):
        import warnings

        warnings.warn("\nFiltered NaN values.", RuntimeWarning)

    x_values = x_values[nan_mask]
    y_values = y_values[nan_mask]
    x_err = x_err[nan_mask]
    y_err = y_err[nan_mask]

    linestyle = kwargs.pop("linestyle", "none")
    zorder = kwargs.pop("zorder", 3)

    scatter_artist = ax.scatter(x_values, y_values, zorder=zorder, **kwargs)
    face_colors = scatter_artist.get_facecolor()

    errorbar = None
    if with_error and (np.any(x_err > 0) or np.any(y_err > 0)):
        errorbar = ax.errorbar(
            x_values,
            y_values,
            xerr=x_err,
            yerr=y_err,
            linestyle=linestyle,
            color=face_colors,
            zorder=zorder + 0.1,
        )

    if not change_viewport:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return ScatterResult(
        scatter=scatter_artist,
        errorbar=errorbar,
        plot=plot,
    )


def scatter(data: object | None = None, *args: object, **kwargs: Any) -> ScatterResult:
    if data is None and "x" in kwargs and "y" in kwargs:
        x_values = kwargs.pop("x")
        y_values = kwargs.pop("y")
        return _scatter_dispatch(x_values, y_values, **kwargs)

    return _scatter_dispatch(data, *args, **kwargs)


@singledispatch
def _scatter_dispatch(data: object, *args: object, **kwargs: Any) -> ScatterResult:
    if len(args) != 1:
        raise TypeError("graph.scatter() expects x/y values or a DataCluster.")

    target_plot = resolve_plot(kwargs.pop("plot", None))
    return _scatter_xy(
        cast(Sequence[SupportedValues] | np.ndarray, data),
        cast(Sequence[SupportedValues] | np.ndarray, args[0]),
        plot=target_plot,
        **kwargs,
    )


@_scatter_dispatch.register
def _(
    data: DataCluster,
    *args: object,
    x: str | None = None,
    y: str | None = None,
    x_index: str | None = None,
    y_index: str | None = None,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult:
    if len(args) == 2:
        if x is not None or y is not None or x_index is not None or y_index is not None:
            raise TypeError("Use either positional x/y indices or x_index/y_index, not both.")
        x_index, y_index = str(args[0]), str(args[1])
    elif len(args) == 0:
        x_index = x if x_index is None else x_index
        y_index = y if y_index is None else y_index
    else:
        raise TypeError("graph.scatter(DataCluster) expects x and y column indices.")

    if x_index is None or y_index is None:
        raise TypeError("graph.scatter(DataCluster) requires x and y column indices.")

    target_plot = resolve_plot(plot)
    if str(x_index) not in data.get_column_names():
        raise ValueError(f'Column "{x_index}" not found in data.')
    if str(y_index) not in data.get_column_names():
        raise ValueError(f'Column "{y_index}" not found in data.')

    return _scatter_xy(
        data.column(str(x_index)),
        data.column(str(y_index)),
        plot=target_plot,
        x_interval=x_interval,
        with_error=with_error,
        change_viewport=change_viewport,
        warn_filter_nan=warn_filter_nan,
        **kwargs,
    )


@_scatter_dispatch.register
def _(
    data: pd.DataFrame,
    *args: object,
    x: str | None = None,
    y: str | None = None,
    x_index: str | None = None,
    y_index: str | None = None,
    plot: Plot | None = None,
    x_interval: tuple[float, float] | None = None,
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any,
) -> ScatterResult:
    if len(args) == 2:
        if x is not None or y is not None or x_index is not None or y_index is not None:
            raise TypeError("Use either positional x/y indices or x_index/y_index, not both.")
        x_index, y_index = str(args[0]), str(args[1])
    elif len(args) == 0:
        x_index = x if x_index is None else x_index
        y_index = y if y_index is None else y_index
    else:
        raise TypeError("graph.scatter(DataFrame) expects x and y column indices.")

    if x_index is None or y_index is None:
        raise TypeError("graph.scatter(DataFrame) requires x and y column indices.")

    target_plot = resolve_plot(plot)
    return _scatter_xy(
        dataframe_column(data, str(x_index)).to_numpy(dtype=object),
        dataframe_column(data, str(y_index)).to_numpy(dtype=object),
        plot=target_plot,
        x_interval=x_interval,
        with_error=with_error,
        change_viewport=change_viewport,
        warn_filter_nan=warn_filter_nan,
        **kwargs,
    )


__all__ = ["scatter"]
