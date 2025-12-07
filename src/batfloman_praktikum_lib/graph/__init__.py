from .quickMethods import create_plot, save_plot, plt
from .plotNScatter import plot, plot_func, plot_line_at_point, scatter, scatter_data, plot_data, filter_nan_values
from .types import PlotResult, ScatterResult, SupportedValues

__all__ = [
    "plt", # expose plt, so we dont need to include it if we need to change somthing deeper (e.g. plt.tight_layout())
    "create_plot", 
    "save_plot", # also creates the dir if necessary
    "filter_nan_values",

    "SupportedValues",
    "PlotResult",
    "plot",
    "plot_func",
    "plot_line_at_point",
    "plot_data",

    "ScatterResult",
    "scatter",
    "scatter_data", 
]
