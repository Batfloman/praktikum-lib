from typing import Sequence, Tuple, Union, List, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from .types import SupportedValues, ScatterResult, PlotResult

def filter_nan_values(
    x: Union[Sequence[SupportedValues], np.ndarray],
    y: Union[Sequence[SupportedValues], np.ndarray],
    warn_filter_nan: bool = True
) -> Tuple[list[float], list[float]]:
    """
    Remove pairs of values where either x or y is NaN.

    Parameters
    ----------
    x : Sequence[SupportedValues]
        Sequence of numeric values.
    y : Sequence[SupportedValues]
        Sequence of numeric values, same length as x.
    warn_filter_nan : bool, default True
        If True, issue a RuntimeWarning when NaNs are filtered.

    Returns
    -------
    Tuple[list[float], list[float]]
        Cleaned x and y with NaNs removed.
    """
    ...

# ==================================================
# plot
# ==================================================

# ==================================================
# scatter
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
    """
    Plot a scatter plot of x vs y with optional error bars (depending on whether x,y are `Measurement`s) .

    Parameters
    ----------
    x : List[SupportedValues]
        X-axis values.
    y : List[SupportedValues]
        Y-axis values, same length as x.
    plot : Tuple[Figure, Axes]
        A matplotlib Figure and Axes object to draw on.
    with_error : bool, default True
        Whether to plot error bars.
    change_viewport : bool, default True
        If False, the current axis limits are restored after plotting.
    warn_filter_nan : bool, default True
        If True, warns when NaNs are filtered from the data.
    **kwargs : Any
        Additional keyword arguments are passed to ax.scatter.

    Returns
    -------
    ScatterResult
        Object containing the scatter plot object and the optional errorbar.
    """
    ...

def scatter_data(
    data: DataCluster,
    x_index: str,
    y_index: str,
    plot: Tuple[Figure, Axes],
    with_error: bool = True,
    change_viewport: bool = True,
    warn_filter_nan: bool = True,
    **kwargs: Any
) -> ScatterResult:
    """
    Plot scatter data from a DataCluster object.

    Parameters
    ----------
    data : DataCluster
        Data source containing columns of numeric values.
    x_index : str
        Column name for x-axis values.
    y_index : str
        Column name for y-axis values.
    plot : Tuple[Figure, Axes]
        Matplotlib figure and axes to draw on.
    with_error : bool, default True
        If True, plot error bars if available.
    change_viewport : bool, default True
        If False, restores original axis limits after plotting.
    warn_filter_nan : bool, default True
        If True, warns when NaNs are filtered from the data.
    **kwargs : Any
        Additional keyword arguments passed to the scatter function.

    Returns
    -------
    ScatterResult
        Object containing scatter and optional errorbar.
    """
    ...

