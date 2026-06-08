from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..path_managment import PathInput

AxesArray = np.ndarray[Any, np.dtype[object]]
SubplotsAxes = Axes | AxesArray
Plot: TypeAlias = tuple[Figure, SubplotsAxes]
ShowArg: TypeAlias = Figure | Plot | Iterable["ShowArg"]

def current_plot() -> Plot | None: ...
def set_current_plot(plot: Plot | None) -> Plot | None: ...
def resolve_plot(plot: Plot | None = None) -> Plot: ...

def save_plot(
    plot: tuple[Figure, SubplotsAxes],
    path: PathInput,
    *,
    print_successMsg: bool = True,
    auto_create_dir: bool = False,
) -> None: ...

@overload
def create_plot(*, squeeze: Literal[False], **kwargs: Any) -> tuple[Figure, AxesArray]: ...

@overload
def create_plot(*, squeeze: bool = True, **kwargs: Any) -> tuple[Figure, SubplotsAxes]: ...

def show(*plots: ShowArg, ignore_quiet: bool = False) -> None: ...
