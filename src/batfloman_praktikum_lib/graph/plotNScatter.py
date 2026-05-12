import numpy as np;
import warnings

# typing
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union, NamedTuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from ..graph_fit.fitResult import FitResult

from .adapter_measurement import extract_value_error

from .types import SupportedValues, ScatterResult, PlotResult

type ErrorBandMode = Union[bool, Literal["auto"]]

def _fit_quality_label(
    fit_result: FitResult,
    *,
    decimals: int = 4,
    decimal_comma: bool = False,
) -> str:
    quality = f"{fit_result.quality:.{decimals}f}"
    if decimal_comma:
        quality = quality.replace(".", r"{,}")

    if fit_result.method == "ODR":
        return fr"$\chi^2_{{\mathrm{{red, ODR}}}} = {quality}$"
    return fr"$\chi^2_\mathrm{{red}} = {quality}$"

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

def filter_nan_values(
    x: Union[Sequence[SupportedValues], np.ndarray],
    y: Union[Sequence[SupportedValues], np.ndarray],
    warn_filter_nan: bool = True
):
    x_arr = np.array(x, dtype=float);
    y_arr = np.array(y, dtype=float);

    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_clean = x[mask]
    y_clean = y[mask]

    if warn_filter_nan and len(x) != len(x_clean):
        warnings.warn("\nFiltered NaN values.", RuntimeWarning)

    return x_clean, y_clean

# ==================================================

def plot(
    x: Sequence[SupportedValues],
    y: Sequence[SupportedValues],
    plot: Tuple[Figure, Any],
    change_viewport: bool =True,
    with_error: bool = True,
    **kwargs,
) -> PlotResult:
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")

    _, ax = plot

    # get the current plot dimensions
    xmin, xmax = ax.get_xlim();
    ymin, ymax = ax.get_ylim();

    # extract values
    x_values, x_err = extract_value_error(x)
    y_values, y_err = extract_value_error(y)

    # plot line
    zorder = kwargs.pop("zorder", 3)
    line, = ax.plot(x_values, y_values, zorder=zorder, **kwargs);

    # plot error-area
    fill = None
    if with_error and np.any(y_err > 0):
        y_min = y_values - y_err
        y_max = y_values + y_err
        fill = ax.fill_between(
            x_values, y_min, y_max,
            color=line.get_color(),
            alpha=0.3,
            label=fr"{line.get_label()} $\pm 1 \sigma$" if line.get_label() else None,
            zorder=zorder-.1
        )

    # keep the same viewport after plotting
    if not change_viewport:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return PlotResult(line=line, fill=fill);

plot_xy = plot # access point for plot function; allowing the `plot` parameter-name in functions

def plot_data(
    data: DataCluster,
    x_index: str,
    y_index: str,
    plot: Tuple[Figure, Any],
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs
) -> PlotResult:
    if not x_index in data.get_column_names():
        raise ValueError(f"Column \"{x_index}\" not found in data.")
    if not y_index in data.get_column_names():
        raise ValueError(f"Column \"{y_index}\" not found in data.")

    x = data.column(x_index)
    y = data.column(y_index)

    x, y = filter_nan_values(x, y, warn_filter_nan=warn_filter_nan)

    return plot_xy(
        x=x,
        y=y,
        plot=plot,
        with_error=with_error,
        change_viewport=change_viewport,
        **kwargs
    );


# ==================================================

def plot_func(
    fit_func: Union[Callable, FitResult],
    plot: Tuple[Figure, Any],
    interval: Optional[Tuple[float, float]] = None,
    change_viewport: bool =True,
    with_error: ErrorBandMode = "auto",
    log_scale: bool = False,
    min_error_band_fraction: float = 0.05,
    show_fit_quality_label: bool = True,
    fit_quality_label_decimal_comma: bool = True,
    fit_quality_label_decimals: int = 4,
    **kwargs
) -> PlotResult:
    _, ax = plot;
    if min_error_band_fraction < 0:
        raise ValueError("min_error_band_fraction must be non-negative.")
    if with_error != "auto" and not isinstance(with_error, bool):
        raise ValueError('with_error must be True, False, or "auto".')

    xmin, xmax = ax.get_xlim() if not interval else interval;

    if log_scale:
        xmin = np.log10(max(xmin, 1e-10)) # kleine Verschiebung, um log(0) zu vermeiden
        xmax = np.log10(xmax)

    x_smooth = np.linspace(xmin, xmax, 10000)

    if log_scale:
        x_smooth = 10**x_smooth 
    x_smooth = list(x_smooth)

    func = fit_func.func if isinstance(fit_func, FitResult) else fit_func
    y_smooth = [func(x) for x in x_smooth]

    if with_error == "auto":
        with_error = (
            _auto_show_fit_error_band(
                y_smooth,
                ax,
                min_error_band_fraction=min_error_band_fraction,
            )
            if isinstance(fit_func, FitResult)
            else True
        )

    if (
        isinstance(fit_func, FitResult)
        and show_fit_quality_label
        and "label" not in kwargs
    ):
        kwargs["label"] = _fit_quality_label(
            fit_func,
            decimals=fit_quality_label_decimals,
            decimal_comma=fit_quality_label_decimal_comma,
        )

    return plot_xy(x_smooth, y_smooth, plot=plot, change_viewport=change_viewport, with_error=with_error, **kwargs)

def plot_line_at_point(
    m: SupportedValues,
    point: Tuple[SupportedValues, SupportedValues],
    plot: Tuple[Figure, Any],
    interval: Optional[Tuple[float, float]] = None,
    change_viewport: bool =True,
    with_error: bool = True,
    log_scale: bool = False,
    **kwargs
) -> PlotResult:
    """
    Plot a line with given slope `m` at a point (x,y).

    Intented use: tangent lines.

    Parameters:
    fit_func (callable): The fitted model function with specific parameters.
    plot (tuple): Tuple containing the figure and axis objects for plotting.
    """
    b = point[1] - m * point[0]
    def func(x):
        return m * x + b

    return plot_func(
        func, 
        plot, 
        interval=interval, 
        change_viewport=change_viewport, 
        with_error=with_error, 
        log_scale=log_scale,
        **kwargs
    )

# ==================================================

def scatter(
    x: Union[Sequence[SupportedValues], np.ndarray],
    y: Union[Sequence[SupportedValues], np.ndarray],
    plot: Tuple[Figure, Axes],
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any
) -> ScatterResult:
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")
    x_values, y_values = filter_nan_values(x, y, warn_filter_nan=warn_filter_nan)

    fig, ax = plot

    # get the current plot dimensions
    xmin, xmax = ax.get_xlim();
    ymin, ymax = ax.get_ylim();

    x_values, x_err = extract_value_error(x)
    y_values, y_err = extract_value_error(y)

    linestyle = kwargs.pop("linestyle", "none")
    zorder = kwargs.pop("zorder", 3)
    
    scatter = ax.scatter(x_values, y_values, zorder=zorder, **kwargs)
    face_colors = scatter.get_facecolor()

    errorbar = None
    if with_error and (np.any(x_err > 0) or np.any(y_err > 0)):
        errorbar = ax.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, linestyle=linestyle, color=face_colors, zorder=zorder+.1)

    # restore viewport
    if not change_viewport:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return ScatterResult(
        scatter=scatter,
        errorbar=errorbar
    )

# ==================================================

def scatter_data(
    data: DataCluster,
    x_index: str,
    y_index: str,
    plot: Tuple[Figure, Any],
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs
) -> ScatterResult:
    if not x_index in data.get_column_names():
        raise ValueError(f"Column \"{x_index}\" not found in data.")
    if not y_index in data.get_column_names():
        raise ValueError(f"Column \"{y_index}\" not found in data.")

    return scatter(
        x=data.column(x_index),
        y=data.column(y_index),
        plot=plot,
        with_error=with_error,
        change_viewport=change_viewport,
        warn_filter_nan=warn_filter_nan,
        **kwargs
    );
