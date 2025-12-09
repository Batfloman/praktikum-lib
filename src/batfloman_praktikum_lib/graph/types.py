import numpy as np

from dataclasses import dataclass
from typing import Union as _Union, Optional

from matplotlib.lines import Line2D as _Line2D
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.container import ErrorbarContainer as _ErrorbarContainer

from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase as _MeasurementBase

# ==================================================
# Define a named tuple for the return values

type SupportedValues = _Union[int, float, _MeasurementBase, np.integer, np.floating]

@dataclass
class PlotResult:
    line: _Line2D
    fill: Optional[PolyCollection] = None

    def remove(self):
        self.line.remove()
        if self.fill:
            self.fill.remove()


@dataclass
class ScatterResult:
    scatter: PathCollection
    errorbar: Optional[_ErrorbarContainer] = None

    def remove(self):
        self.scatter.remove()
        if self.errorbar:
            self.errorbar.remove()

__all__ = ["SupportedValues", "PlotResult", "ScatterResult"]
