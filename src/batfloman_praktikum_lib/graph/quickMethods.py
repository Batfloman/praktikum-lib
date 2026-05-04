import matplotlib.pyplot as plt;
import numpy as np;
import os

# typing
from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, overload
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib._pylab_helpers import Gcf

from ..io.termColors import bcolors
from ..path_managment import create_dirs, dir_exist
from ..flags import check_quiet

# ==================================================

AxesArray = np.ndarray[Any, np.dtype[object]]
SubplotsAxes = Axes | AxesArray
Plot: TypeAlias = tuple[Figure, SubplotsAxes]
ShowArg: TypeAlias = Figure | Plot | Iterable["ShowArg"]

def save_plot(
        plot: tuple[Figure, SubplotsAxes],
        path: str,
        *,
        print_successMsg: bool = True,
        auto_create_dir: bool = False,
):
    fig, _ = plot

    if auto_create_dir and not dir_exist(path):
        create_dirs(path)
        print(f"{bcolors.CREATED}Created directory: {bcolors.ENDC}{path}")

    fig.savefig(path, dpi=300)

    if print_successMsg:
        print(f"{bcolors.OKGREEN}Succesfully saved{bcolors.ENDC}",
              f"{bcolors.OKBLUE}Plot{bcolors.ENDC}")
        print(f"\t{bcolors.DIM}to {path}{bcolors.ENDC}")

# --------------------

@overload
def create_plot(*, squeeze: Literal[False], **kwargs) -> tuple[Figure, AxesArray]: ...

@overload
def create_plot(*, squeeze: bool = True, **kwargs) -> tuple[Figure, SubplotsAxes]: ...

def create_plot(*, squeeze: bool = True, **kwargs) -> tuple[Figure, SubplotsAxes]:
    kwargs["squeeze"] = squeeze
    return plt.subplots(**kwargs);

# --------------------

def _is_plot_tuple(plot: object) -> bool:
    return (
        isinstance(plot, tuple)
        and len(plot) == 2
        and isinstance(plot[0], Figure)
    )


def _iter_figures(plots: Iterable[ShowArg]) -> Iterable[Figure]:
    for plot in plots:
        if isinstance(plot, Figure):
            yield plot
            continue

        if _is_plot_tuple(plot):
            fig, _ = plot
            yield fig
            continue

        if isinstance(plot, Iterable) and not isinstance(plot, (str, bytes)):
            yield from _iter_figures(plot)
            continue

        raise TypeError(
            "show() expects Figure instances, create_plot() results, or iterables of them"
        )


def _ensure_registered(fig: Figure) -> None:
    if Gcf.get_fig_manager(fig.number) is not None:
        return

    manager = plt._get_backend_mod().new_figure_manager_given_figure(fig.number, fig)
    Gcf._set_new_active_manager(manager)


def show(*plots: ShowArg, ignore_quiet: bool = False) -> None:
    if check_quiet() and not ignore_quiet:
        return

    if plots:
        for fig in _iter_figures(plots):
            _ensure_registered(fig)

    return plt.show()
