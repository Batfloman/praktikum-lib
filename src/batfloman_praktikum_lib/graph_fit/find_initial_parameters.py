from typing import Optional, Union, List, Callable
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from pathlib import Path
import json
import inspect

from ..tables.validation import ensure_extension
from .fitModel import FitModel

# ==================================================

class ParameterManager:
    slider: Slider;

    slider_value: float

    center: float
    min: float
    max: float

    def __init__(self,
        param: str,
        total_ax, 
        initial_value: float,
        update_callback,
        min: Optional[float] = None,
        max: Optional[float] = None,
        center: Optional[float] = None,
    ):
        self.param = param;
        self.center = center or initial_value
        self.min = min or 0.5 * initial_value
        self.max = max or 1.5 * initial_value

        total_ax.set_xticks([])
        total_ax.set_yticks([])
        total_ax.set_frame_on(False)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        slider_ax = inset_axes(
            total_ax,
            width="60%", height="50%",
            loc='center left'
        )
        # Buttons rechts (je 10% Breite)
        ax_mini = inset_axes(
            total_ax,
            bbox_to_anchor=(0.75, 0, 0.1, 1),
            bbox_transform=total_ax.transAxes,
            width="80%", height="80%",
            # loc="center left"
            loc="center"
        )

        ax_cent = inset_axes(
            total_ax,
            bbox_to_anchor=(0.85, 0, 0.1, 1),
            bbox_transform=total_ax.transAxes,
            width="80%", height="80%",
            # loc="center left"
            loc="center"
        )

        ax_expa = inset_axes(
            total_ax,
            bbox_to_anchor=(0.95, 0, 0.1, 1),
            bbox_transform=total_ax.transAxes,
            width="80%", height="80%",
            loc="center"
        )

        def create_slider(ax, label, valmin, valmax, valinit):
            self.slider = Slider(ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            self.slider.label.set_x(1.02)
            self.slider.label.set_horizontalalignment("left")
            self.slider.valtext.set_x(-0.01)
            self.slider.valtext.set_horizontalalignment("right")

        self.slider_value = initial_value
        create_slider(ax=slider_ax, label=param, valmin=self.min, valmax=self.max, valinit=initial_value)

        def get_delta_text():
           # return f"Δ = {self.max - self.min:.2e}"
            return fr"Range $({self.center:.3g} \pm {abs(self.max - self.min)/2:.2g})$"

        self.text_delta = total_ax.text(
            0.31, 0.9, get_delta_text(), ha="center", va="center",
            transform=total_ax.transAxes
        )
        self.b1 = Button(ax_mini, '–')
        self.b2 = Button(ax_cent, '●')
        self.b3 = Button(ax_expa, '+')

        def slider_callback(val):
            self.slider_value = val;
            update_callback()

        self.slider.on_changed(slider_callback)

        def recreate_slider():
            ax = self.slider.ax
            label = self.slider.label.get_text()
            val = self.slider.val
            ax.clear()
            create_slider(ax=ax, label=label, valmin=self.min, valmax=self.max, valinit=val)
            self.slider.on_changed(slider_callback)  # reattach callback

        def btn_center_callback(event):
            offset = self.slider_value - self.center
            self.min += offset
            self.max += offset
            self.center += offset

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b2.on_clicked(btn_center_callback);

        def btn_minimize_callback(event):
            delta = abs(self.min - self.max)
            new_delta = 0.5 * delta;

            self.min = self.center - 0.5 * new_delta
            self.max = self.center + 0.5 * new_delta

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b1.on_clicked(btn_minimize_callback)

        def btn_expand_callback(event):
            delta = abs(self.min - self.max)
            new_delta = 2 * delta;

            self.min = self.center - 0.5 * new_delta
            self.max = self.center + 0.5 * new_delta

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b3.on_clicked(btn_expand_callback)

    def get_value(self):
        return self.slider_value

    def to_dict(self):
        return {
            "min": self.min,
            "max": self.max,
            "center": self.center,
            "slider_value": self.slider_value,
        }

def order_initial_params(
    model: Union[Callable, FitModel],
    initial_params: Union[List[float], dict[str, float], dict[str, ParameterManager]],
) -> List[float]:
    model_fn = model.model if isinstance(model, FitModel) else model
    # assume the first parameter is the continous
    params = list(inspect.signature(model_fn).parameters.keys())[1:]

    # if list assume the params are in order:
    if isinstance(initial_params, list):
        if len(params) != len(initial_params):
             raise ValueError(
                f"Expected {len(params)} initial parameters, got {len(initial_params)}."
            )

        return initial_params
    # else check & filter to only used parameters
    elif isinstance(initial_params, dict):
        vals = []
        for p in params:
            if p not in initial_params:
                raise ValueError(
                    f"Missing initial parameter: '{p}'. "
                    f"Expected parameters: {params}"
                )
            val = initial_params[p]
            if isinstance(val, ParameterManager):
                vals.append(val.slider_value)
            elif isinstance(val, dict):
                vals.append(val.get("slider_value", 1.0))
            else:
                vals.append(val)
        return vals
    else:
        raise ValueError(
            "initial_params must be either a list of floats or a dict[str, float]."
        )


# ==================================================
# saving & loading

def _save_slider_settings(file: Path, params: dict[str, ParameterManager]):
    serializable = {}
    for k, v in params.items():
        serializable[k] = v.to_dict()
    file.write_text(json.dumps(serializable, indent=2))

def _load_slider_settings(file: Path) -> dict[str, dict]:
    d = {};
    if not file.exists():
        return d;

    return json.loads(file.read_text())

def _extract_default_values(
    x_data,
    y_data,
    model: Union[Callable, FitModel],
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

# ==================================================

def find_init_params(
    x_data,
    y_data,
    model: Union[Callable, FitModel],
    cachePath="fitcache.json",
    default_values: Optional[Union[List[float], dict[str, float]]] = None
) -> dict[str, float]:
    filepath = ensure_extension(cachePath, ".json")
    CACHE = Path(filepath);

    fig_graph, ax_graph = plt.subplots()
    fig_slider, ax_slider = plt.subplots()
    ax_slider.set_xticks([])
    ax_slider.set_yticks([])
    ax_slider.set_frame_on(False)

    # fig, ax = plt.subplots()
    ax_graph.set_title("Find Parameters")
    ax_graph.scatter(x_data, y_data)

    # assume first parameter is the continous parameter (e.g. `x` in `f(x)`)
    params = list(inspect.signature(model).parameters.keys())[1:]

    sliderheight = min(0.075, 1/(len(params)+2))
    # offset = sliderheight * len(params) + 2 * sliderheight
    # fig.subplots_adjust(bottom=offset)
    
    default = _extract_default_values(x_data, y_data, model, default_values)
    default_vals: List[float] = order_initial_params(model, default);
    cached = _load_slider_settings(CACHE)

    cached_values = {};
    for p in params:
        if p in cached:
            settings = cached[p]
            val = settings.get("slider_value", None);
            if val is not None:
                cached_values[p] = val

    starting_params = order_initial_params(model, {**default, **cached_values})

    x_plot = np.linspace(x_data.min(), x_data.max(), 200)
    line, = ax_graph.plot(x_plot, model(x_plot, *starting_params), color='red')

    parameters: dict[str, ParameterManager] = {};

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

        parameters[p] = ParameterManager(
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

    _save_slider_settings(CACHE, parameters)

    return initial_parameters;
