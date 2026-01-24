from typing import Callable, List, Union, Any
from collections.abc import Sequence, Mapping
import inspect


from ..models import FitModel
from .parameterSlider import ParameterSlider
from ._helper import get_model_fn

def _extract_value(v):
    # Wenn das Objekt ein slider_value Attribut hat
    if hasattr(v, "slider_value"):
        return v.slider_value
    # Wenn Mapping
    if isinstance(v, Mapping):
        return v.get("slider_value", 1.0)
    return float(v)


def order_initial_params(
    model: Callable | FitModel,
    initial_params: Union[
        Sequence[float],
        Mapping[str, Any]
    ],
) -> List[float]:
    model_fn = get_model_fn(model)

    # assume first argument is x (independent, continuous variable)
    params = list(inspect.signature(model_fn).parameters)[1:]

    # if list assume the params are in order:
    if isinstance(initial_params, Sequence) and not isinstance(initial_params, (str, bytes, Mapping)):
        if len(params) != len(initial_params):
            raise ValueError(f"Expected {len(params)} initial parameters, got {len(initial_params)}.")
        return list(initial_params)

    # else check & filter to only used parameters
    if isinstance(initial_params, Mapping):
        vals = []
        for p in params:
            if p not in initial_params:
                raise ValueError(
                    f"Missing initial parameter: '{p}'. "
                    f"Expected parameters: {params}"
                )
            vals.append(_extract_value(initial_params[p]))
        return vals

    raise TypeError(
        "initial_params must be either a list of floats or a dict[str, float]."
    )
