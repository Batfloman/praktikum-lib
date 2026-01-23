from typing import Optional, Union, List, Callable, Type
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import inspect
from inspect import isclass

# this lib
from batfloman_praktikum_lib.path_managment import ensure_extension, validate_filename
from ..models import FitModel

# this module
from ._helper import extract_default_values
from .order_init_params import order_initial_params
from .parameterSlider import ParameterSlider, save_slider_settings, load_slider_settings

# ==================================================

def manual_init_params(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    *,
    cache_path= "fitcache.json" ,
    default_values: Optional[Union[List[float], dict[str, float]]] = None,
    warn_filter_nan: bool = True,
    use_cache: bool = False,
) -> dict[str, float]:
    filepath = ensure_extension(cache_path, ".json")
    CACHE = Path(filepath);
    
    # --------------------
    # deconststruct parameters
    # --------------------

    # has_components = False
    # components = {}
    if isclass(model) and issubclass(model, FitModel):
        # has_components = issubclass(model, CompositeFitModel)
        model = model.model
        # TODO: Add drawing individual components

    from batfloman_praktikum_lib.graph.plotNScatter import filter_nan_values
    x_data, y_data = filter_nan_values(x_data, y_data, warn_filter_nan=warn_filter_nan)
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # assume first parameter is the continous parameter (e.g. `x` in `f(x)`)
    params = list(inspect.signature(model).parameters.keys())[1:]

    default = extract_default_values(x_data, y_data, model, default_values)
    default_vals: List[float] = order_initial_params(model, default);
    cached = load_slider_settings(CACHE)

    cached_values = {};
    for p in params:
        if p in cached:
            settings = cached[p]
            val = settings.get("slider_value", None);
            if val is not None:
                cached_values[p] = val

    starting_params = order_initial_params(model, {**default, **cached_values})

    if use_cache:
        if not os.path.exists(filepath):
            raise ReferenceError("File not found!")
        else: 
            return {**default, **cached_values};

    # --------------------
    # plot
    # --------------------

    fig_graph, ax_graph = plt.subplots()
    fig_slider, ax_slider = plt.subplots()
    ax_slider.set_xticks([])
    ax_slider.set_yticks([])
    ax_slider.set_frame_on(False)

    sliderheight = min(0.075, 1/(len(params)+2))

    # fig, ax = plt.subplots()
    ax_graph.set_title("Find Parameters")
    from ...graph import scatter
    scatter(x_data, y_data, plot=(fig_graph, ax_graph), zorder=1)

    xmin, xmax = np.min(x_data), np.max(x_data)
    x_plot = np.linspace(xmin, xmax, 10000)
    line, = ax_graph.plot(x_plot, model(x_plot, *starting_params), color='red', zorder=2)

    parameters: dict[str, ParameterSlider] = {};

    def redraw():
        params = order_initial_params(model, parameters)
        y_line = model(x_plot, *params)
        line.set_ydata(y_line)

        # Achsen automatisch anpassen
        ymin = min(y_data.min(), np.min(y_line))
        ymax = max(y_data.max(), np.max(y_line))
        dy = ymax - ymin or 1.0
        pad = 0.05 * dy
        ax_graph.set_ylim(ymin - pad, ymax + pad)

        fig_graph.canvas.draw_idle()

    for i, p in enumerate(params):
        settings = cached[p] if p in cached else {};
        val = settings.get("slider_value")
        initial_value = val if val is not None else default_vals[i];

        parameters[p] = ParameterSlider(
            param = p,
            # total_ax = fig_slider.add_axes((0.10, offset - (i+2)*sliderheight, 0.8, sliderheight)),
            total_ax = fig_slider.add_axes((0.1, 1 - (i+2)*sliderheight, 0.85, sliderheight)),
            initial_value = initial_value,
            update_callback = redraw,
            min = settings.get("min", None), 
            max = settings.get("max", None), 
            center = settings.get("center", None), 
        )

    plt.show()

    initial_parameters = {
        p: v for p, v in zip(params, order_initial_params(model, parameters))
    }

    save_slider_settings(CACHE, parameters)

    return initial_parameters;
