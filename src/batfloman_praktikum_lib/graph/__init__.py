from .quickMethods import create_plot, save_plot, plt
from .plotNScatter import plot, plot_func, plot_line_at_point, scatter, scatter_data
from .types import PlotResult, ScatterResult, SupportedValues

__all__ = [
    "plt", # expose plt, so we dont need to include it if we need to change somthing deeper (e.g. plt.tight_layout())
    "create_plot", 
    "save_plot", # also creates the dir if necessary

    "SupportedValues",
    "PlotResult",
    "plot",
    "plot_func",
    "plot_line_at_point",

    "ScatterResult",
    "scatter",
    "scatter_data", 
]
