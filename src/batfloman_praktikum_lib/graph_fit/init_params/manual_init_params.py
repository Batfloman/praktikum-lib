from typing import Optional, Union, List, Callable, Type
import numpy as np
from pathlib import Path
import inspect
from inspect import isclass
import json

from PyQt6.QtWidgets import QApplication
from .parameterWindow import ParameterWindow
from .graphWindow import GraphWindow
from .parameterSlider import ParameterSlider, save_slider_settings, load_slider_settings
from ..models import FitModel
from ._helper import extract_default_values, get_model_fn
from .order_init_params import order_initial_params

def manual_init_params(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    *,
    cache_path="fitcache.json",
    default_values: Optional[Union[List[float], dict[str, float]]] = None,
    warn_filter_nan: bool = True,
    use_cache: bool = False,
) -> dict[str, float]:
    # --------------------
    # cache
    filepath = Path(cache_path).with_suffix(".json")
    cached = load_slider_settings(filepath)

    # --------------------
    # Unpack model & data
    model = get_model_fn(model)

    from batfloman_praktikum_lib.graph.plotNScatter import filter_nan_values
    x_data, y_data = filter_nan_values(x_data, y_data, warn_filter_nan=warn_filter_nan)
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # --------------------
    # Parameters
    param_names = list(inspect.signature(model).parameters.keys())[1:]
    default_dict = extract_default_values(x_data, y_data, model, default_values)

    # load cached slider values if present
    cached_values = {
        p: cached[p]["slider_value"]
        for p in param_names
        if p in cached and "slider_value" in cached[p]
    }

    starting_params = order_initial_params(model, {**default_dict, **cached_values})

    if use_cache:
        return {**default_dict, **cached_values}

    # --------------------
    # Start Qt app
    app = QApplication.instance() or QApplication([])

    # --------------------
    # Create Graph Window
    graph_win = GraphWindow(
        x_data=x_data,
        y_data=y_data,
        model=model,
        params=dict(zip(param_names, starting_params))
    )

    # --------------------
    # Callback for slider updates
    def update_graph():
        params = {name: s.get_value() for name, s in param_win.sliders.items()}
        graph_win.update_params(params)

    # --------------------
    # Create Parameter Window
    # param_win = ParameterWindow(
    #     params=dict(zip(param_names, starting_params)),
    #     update_callback=update_graph
    # )
    param_win = ParameterWindow(
        params={
            name: ParameterSlider.from_cache(name, cached, default)
            for name, default in zip(param_names, starting_params)
        },
        update_callback=update_graph
    )

    # --------------------
    # bind the windows, so user closes both at the same time
    graph_win.param_win = param_win
    param_win.graph_win = graph_win

    graph_win.show()
    param_win.show()

    # --------------------
    # Execute Qt loop
    app.exec()

    # --------------------
    # Save final slider settings
    save_slider_settings(filepath, param_win.sliders)

    return param_win.get_params()
