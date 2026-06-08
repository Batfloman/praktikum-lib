import weakref
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib._pylab_helpers import Gcf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..flags import should_skip_popup_sequence
from ..io.termColors import bcolors
from ..path_managment import PathInput, create_dirs, dir_exist

AxesArray = np.ndarray[Any, np.dtype[object]]
SubplotsAxes = Axes | AxesArray
Plot: TypeAlias = tuple[Figure, SubplotsAxes]
ShowArg: TypeAlias = Figure | Plot | Iterable["ShowArg"]

_tracked_figures: weakref.WeakValueDictionary[int, Figure] = weakref.WeakValueDictionary()
_current_plot: Plot | None = None


def _track_figure(fig: Figure) -> Figure:
    _tracked_figures[fig.number] = fig
    return fig


def current_plot() -> Plot | None:
    return _current_plot


def set_current_plot(plot: Plot | None) -> Plot | None:
    global _current_plot

    if plot is not None:
        fig, _ = plot
        _track_figure(fig)

    _current_plot = plot
    return plot


def resolve_plot(plot: Plot | None = None) -> Plot:
    if plot is not None:
        set_current_plot(plot)
        return plot

    resolved_plot = current_plot()
    return create_plot() if resolved_plot is None else resolved_plot


def save_plot(
    plot: tuple[Figure, SubplotsAxes],
    path: PathInput,
    *,
    print_successMsg: bool = True,
    auto_create_dir: bool = False,
) -> None:
    fig, _ = plot

    if auto_create_dir and not dir_exist(path):
        create_dirs(path)
        print(f"{bcolors.CREATED}Created directory: {bcolors.ENDC}{path}")

    fig.savefig(path, dpi=300)

    if print_successMsg:
        print(f"{bcolors.OKGREEN}Succesfully saved{bcolors.ENDC}",
              f"{bcolors.OKBLUE}Plot{bcolors.ENDC}")
        print(f"\t{bcolors.DIM}to {path}{bcolors.ENDC}")


@overload
def create_plot(*, squeeze: Literal[False], **kwargs: Any) -> tuple[Figure, AxesArray]: ...


@overload
def create_plot(*, squeeze: bool = True, **kwargs: Any) -> tuple[Figure, SubplotsAxes]: ...


def create_plot(*, squeeze: bool = True, **kwargs: Any) -> tuple[Figure, SubplotsAxes]:
    kwargs["squeeze"] = squeeze
    fig, axes = plt.subplots(**kwargs)
    _track_figure(fig)
    return set_current_plot((fig, axes))


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
    _track_figure(fig)

    if Gcf.get_fig_manager(fig.number) is not None:
        return

    manager = plt._get_backend_mod().new_figure_manager_given_figure(fig.number, fig)
    Gcf._set_new_active_manager(manager)


def _show_registered(figures: Iterable[Figure]) -> None:
    selected_numbers = {fig.number for fig in figures}
    hidden_managers = OrderedDict(
        (num, manager)
        for num, manager in tuple(Gcf.figs.items())
        if num not in selected_numbers
    )

    for num in hidden_managers:
        Gcf.figs.pop(num, None)

    try:
        plt.show(block=True)
    finally:
        for num, manager in hidden_managers.items():
            if Gcf.get_fig_manager(num) is None:
                Gcf.figs[num] = manager


def show(*plots: ShowArg, ignore_quiet: bool = False) -> None:
    if not ignore_quiet and should_skip_popup_sequence():
        return

    if not plots:
        return plt.show(block=True)

    figures = tuple(_iter_figures(plots))
    for fig in figures:
        _ensure_registered(fig)

    _show_registered(figures)
