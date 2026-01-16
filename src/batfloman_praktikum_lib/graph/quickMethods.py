import matplotlib.pyplot as plt;
import numpy as np;
import os

# typing
from typing import Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..io.termColors import bcolors
from ..path_managment import create_dirs, dir_exist
from ..flags import check_quiet

# ==================================================

def save_plot(
        plot: tuple[Figure, Axes | np.ndarray],
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

def create_plot(**kwargs) -> tuple[Figure, Any]:
    return plt.subplots(**kwargs);

# --------------------

def show(*, ignore_quiet: bool = False) -> None:
    if check_quiet() and not ignore_quiet:
        return
    return plt.show()
