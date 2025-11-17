import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd
import os

# typing
from typing import Any, Callable, List, Optional, Tuple, Union, NamedTuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
from batfloman_praktikum_lib.graph_fit import FitResult

# Define a named tuple for the return value
class PlotResult(NamedTuple):
    line: Line2D
    plot: tuple[Figure, Any]
    fill: Optional[PolyCollection]

class ScatterResult(NamedTuple):
    scatter: PathCollection
    plot: tuple[Figure, Any]
    errorbar: Optional[PolyCollection]

def _normalize_column_name(name):
    return name.replace(" ", "")

def _extract_value_error(lst: List[MeasurementBase | float | int | str]):
    values = []
    errors = []
    
    for item in lst:
        if isinstance(item, np.str_):
            try:
                item = float(item) if '.' in item else int(item)
            except:
                raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")

        if isinstance(item, (float, int, np.int64)):
            values.append(item)
            errors.append(0)
        elif isinstance(item, (MeasurementBase)):
            values.append(item.value)
            errors.append(item.error)
        else:
            raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")
    
    return np.array(values), np.array(errors)

def save_plot(plot: tuple[Figure, Axes | np.ndarray], path: str, successMsg: bool = True):
    fig, _ = plot
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(path, dpi=300)
    if successMsg:
        print(f"file {os.path.basename(path)} has been saved to {os.path.abspath(dir_path or '.')}")

def create_plot(**kwargs) -> tuple[Figure, Any]:
    return plt.subplots(**kwargs);

def plot(
    x: List[MeasurementBase | float | int],
    y: List[MeasurementBase | float | int],
    plot: Optional[Tuple[Figure, Any]] = None,
    change_viewport: bool =True,
    with_error: bool = True,
    **kwargs,
) -> PlotResult:
    fig, ax = plot if plot else plt.subplots()

    # get the current plot dimensions
    xmin, xmax = ax.get_xlim();
    ymin, ymax = ax.get_ylim();

    # extract values
    x_values, x_err = _extract_value_error(x)
    y_values, y_err = _extract_value_error(y)

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

    return PlotResult(line=line, plot=(fig, ax), fill=fill);

plot_xy = plot # access point for plot function; allowing the plot parameter in functions

def plot_func(
    fit_func: Union[Callable, FitResult],
    plot: Optional[Tuple[Figure, Any]] = None,
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
    plot = plot if (plot is not None) else plt.subplots();
    _, ax = plot

    # get the current plot dimensions
    xmin, xmax = ax.get_xlim();

    if log_scale:
        # kleine Verschiebung, um log(0) zu vermeiden
        x_smooth = np.logspace(np.log10(max(xmin, 1e-10)), np.log10(xmax), 10000)
    else:
        x_smooth = np.linspace(xmin, xmax, 10000)

    func = fit_func.func if isinstance(fit_func, FitResult) else fit_func
    y_smooth = [func(x) for x in x_smooth]

    return plot_xy(x_smooth, y_smooth, plot=plot, change_viewport=change_viewport, with_error=with_error, **kwargs)

    # min_func = None
    # max_func = None
    # if isinstance(fit_func, FitResult):
    #     min_func = fit_func.min_1sigma;
    #     max_func = fit_func.max_1sigma;
    #     fit_func = fit_func.func_no_err;
    #
    # fig, ax = plot if (plot is not None) else plt.subplots();
    #
    # # get the current plot dimensions
    # xmin, xmax = ax.get_xlim();
    # ymin, ymax = ax.get_ylim();
    #
    # # Generate a smooth line for the model
    # x_smooth = np.linspace(xmin, xmax, 10000)
    # y_smooth = [fit_func(x) for x in x_smooth]
    #
    # # if results are measurements
    # if any(isinstance(y, MeasurementBase) for y in y_smooth):
    #     # Min- und Max-Funktion definieren
    #     def min_func(x):
    #         m = fit_func(x)
    #         return m.value - m.error if isinstance(m, MeasurementBase) else m
    #     def max_func(x):
    #         m = fit_func(x)
    #         return m.value + m.error if isinstance(m, MeasurementBase) else m
    #     # y_smooth auf Werte reduzieren
    #     y_smooth = np.array([y.value if isinstance(y, MeasurementBase) else y for y in y_smooth])
    #
    # zorder = kwargs.pop("zorder", 3)
    # line, = ax.plot(x_smooth, y_smooth, zorder=zorder, **kwargs)
    #
    # # if min, max parameter are provided color the 1-sigma area
    # fill = None
    # if with_error and min_func is not None and max_func is not None:
    #     y_min = [min_func(x) for x in x_smooth]
    #     y_max = [max_func(x) for x in x_smooth]
    #     fill = ax.fill_between(x_smooth, y_min, y_max, color=line.get_color(), alpha=0.3, label=r"$\pm 1 \sigma$", zorder=zorder-.1)
    #
    # # keep the same viewport after plotting
    # if not change_viewport:
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #
    # return PlotResult(line=line, plot=(fig, ax), fill=fill);

def scatter(
    x: List[Measurement | float | int],
    y: List[Measurement | float | int],
    with_error = True,
    plot: Optional[Tuple[Figure, Any]] = None,
    change_viewport: bool = True,
    **kwargs
) -> ScatterResult:
    if len(x) != len(y):
        raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")

    fig, ax = plot if plot else plt.subplots();

    # get the current plot dimensions
    xmin, xmax = ax.get_xlim();
    ymin, ymax = ax.get_ylim();

    x_values, x_err = _extract_value_error(x)
    y_values, y_err = _extract_value_error(y)

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

    return ScatterResult(scatter=scatter, plot=(fig, ax), errorbar=errorbar)

def scatter_data(
    data: DataCluster,
    x_index: str,
    y_index: str,
    with_error: bool = True,
    plot: Optional[Tuple[Figure, Any]] = None,
    change_viewport: bool = True,
    **kwargs
) -> ScatterResult:
    if not x_index in data.get_column_names():
        raise ValueError(f"Column \"{x_index}\" not found in data.")
    if not y_index in data.get_column_names():
        raise ValueError(f"Column \"{y_index}\" not found in data.")

    x = data.column(x_index);
    y = data.column(y_index);
    return scatter(x, y, with_error, plot=plot, change_viewport=change_viewport, **kwargs);

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
    # x = data[x_index]
    # y = data[y_index]
    return scatter(x, y, with_error, plot=plot, **kwargs)


# def lineplot(m, b, start, end, p=None, color=None, label=None):
#     x = np.linspace(start, end, 10000);
#     y = m * x + b;
#     return plot(x, y, p=p, color=color, label=label)

# def lineplot_around_sp(m: float, sp: Point, start, end, p=None, color=None, label=None):
#     x = np.linspace(start, end, 10000);
#     b = calc_b_lineplot(m, sp);
#     y = m * x + b;
#     return plot(x, y, p=p, color=color, label=label)

# def calc_b_lineplot(m, sp):
#     return sp.y - m*sp.x
