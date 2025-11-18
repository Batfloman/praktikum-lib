import matplotlib.pyplot as plt;
import numpy as np;
import os

# typing
from typing import Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# ==================================================

def save_plot(
        plot: tuple[Figure, Axes | np.ndarray],
        path: str,
        print_successMsg: bool = True
):
    fig, _ = plot
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(path, dpi=300)
    if print_successMsg:
        print(f"file {os.path.basename(path)} has been saved to {os.path.abspath(dir_path or '.')}")

# --------------------

def create_plot(**kwargs) -> tuple[Figure, Any]:
    return plt.subplots(**kwargs);

