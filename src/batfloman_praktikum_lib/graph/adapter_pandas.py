from typing import Any

import pandas as pd

from .scatter_dispatch import scatter
from .types import ScatterResult


def scatter_dataframe(
    data: pd.DataFrame,
    x_index: str,
    y_index: str,
    with_error: bool = True,
    plot=None,
    **kwargs: Any,
) -> ScatterResult:
    return scatter(data, x_index, y_index, with_error=with_error, plot=plot, **kwargs)
