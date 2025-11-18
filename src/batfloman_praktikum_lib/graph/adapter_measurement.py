import numpy as np

from functools import singledispatch
from typing import List, Tuple, Sequence

from batfloman_praktikum_lib.graph_fit.fitResult import FitResult

from ..structs.measurementBase import MeasurementBase
from .types import SupportedValues

def extract_value_error(
        lst: Sequence[SupportedValues]
) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    errors = []
    
    for item in lst:
        if isinstance(item, np.str_):
            try:
                item = float(item) if '.' in item else int(item)
            except:
                raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")

        if isinstance(item, (float, int, np.integer)):
            values.append(item)
            errors.append(0)
        elif isinstance(item, (MeasurementBase)):
            values.append(item.value)
            errors.append(item.error)
        else:
            raise ValueError(f"List contains an unsupported type: {item} is type {type(item)}")
    
    return np.array(values), np.array(errors)


