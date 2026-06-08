from typing import Sequence

import numpy as np
import pandas as pd

from .types import SupportedValues

def dataframe_column(data: pd.DataFrame, index: str) -> pd.Series: ...

def extract_value_error(
    values: Sequence[SupportedValues],
) -> tuple[np.ndarray, np.ndarray]: ...

def filter_nan_values(
    x: Sequence[SupportedValues] | np.ndarray,
    y: Sequence[SupportedValues] | np.ndarray,
    warn_filter_nan: bool = True,
): ...
