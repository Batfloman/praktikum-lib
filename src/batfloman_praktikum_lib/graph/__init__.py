from .plot_state import create_plot, current_plot, resolve_plot, save_plot, set_current_plot, plt, show
from .plot_dispatch import fit_quality_label, plot
from .scatter_dispatch import scatter
from .helpers import filter_nan_values
from .types import PlotResult, ScatterResult, SupportedValues

__all__ = [
    "plt", # expose plt, so we dont need to include it if we need to change somthing deeper (e.g. plt.tight_layout())
    "create_plot", 
    "current_plot",
    "resolve_plot",
    "set_current_plot",
    "save_plot", # also creates the dir if necessary
    "filter_nan_values",
    "fit_quality_label",
    "show",

    "SupportedValues",
    "PlotResult",
    "plot",

    "ScatterResult",
    "scatter",
]
