import numpy as np;
import warnings

# typing
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, NamedTuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from ..graph_fit.fitResult import FitResult

from .adapter_measurement import extract_value_error

from .types import SupportedValues, ScatterResult, PlotResult

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
    with_error: bool = True,
    log_scale: bool = False,
    **kwargs
) -> PlotResult:
    """
    Plot data with error bars and a fitted model.

    Parameters:
    fit_func (callable): The fitted model function with specific parameters.
    plot (tuple): Tuple containing the figure and axis objects for plotting.
    """
    _, ax = plot;
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
