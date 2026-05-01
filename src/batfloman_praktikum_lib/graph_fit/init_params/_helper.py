import inspect
from typing import Union, Callable, Optional, List

from ..models import FitModel

def get_model_fn(model):
    from inspect import isclass
    if isinstance(model, FitModel):
        return model.model
    if isclass(model) and issubclass(model, FitModel):
        return model.model
    return model

def smart_format(val: float, sig: int = 3) -> str:
    if val == 0:
        return "0"
    if abs(val) >= 1e5 or abs(val) < 1e-2:
        return f"{val:.{sig}e}"
    if abs(val) >= 1e3:
        return f"{val:.0f}"
    else:
        return f"{val:.{sig}g}"

def extract_default_values(
    x_data,
    y_data,
    model: Callable,
    user_specified: Optional[Union[List[float], dict[str, float]]] = None
) -> dict[str, float]:
    params = list(inspect.signature(model).parameters.keys())[1:]

    # ensure all params have some kind of default
    default = { 
        p: 1.0 for p in params 
    }

    from_guess = {} 
    if isinstance(model, FitModel):
        initial_guess = model.get_initial_guess(x_data, y_data);
        from_guess = { 
            p: v for p, v in zip(params, initial_guess)
        };
        default = {**default, **from_guess}

    if not user_specified:
        return default

    if isinstance(user_specified, dict):
        return {**default, **user_specified}
    if isinstance(user_specified, List):
        if len(params) != len(user_specified):
            raise ValueError("")

        from_values = {
            p: user_specified[i] for i, p in enumerate(params)
        }
        return {**default, **from_values}
