from dataclasses import dataclass, asdict
from typing import Optional, Any, Union, List, Callable, TypedDict
from PIL.Image import init
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np

from pathlib import Path
import json
import inspect
import time

from ..tables.validation import ensure_extension
from .fitModel import FitModel

last_update_time = 0

class InitialParameterType(TypedDict):
    value: float

def order_initial_params(
    model: Union[Callable, FitModel],
    initial_params: Union[List[float], dict[str, float]],
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
            vals.append(initial_params[p])
        return vals
    else:
        raise ValueError(
            "initial_params must be either a list of floats or a dict[str, float]."
        )

# ==================================================
# just types

# @dataclass
# class InitialParameter:
#     value: float
#     name: Optional[str] = None
#
#     @staticmethod
#     def from_any(obj: Any) -> "InitialParameter":
#         if isinstance(obj, InitialParameter):
#             return obj
#         if isinstance(obj, SliderSettings):
#             return InitialParameter(
#                 name = obj.name,
#                 value = obj.value
#             )
#         if isinstance(obj, dict):
#             return InitialParameter(
#                 name = obj.get("name", None),
#                 value = obj.get("value", 1.0),
#             );
#         if isinstance(obj, (int, float, np.integer, np.floating)):
#             return InitialParameter(
#                 value = float(obj),
#             ) 
#
#         raise ValueError("")
        

@dataclass
class SliderSettings:
    slider_value: float
    exponent: int = 0;
    name: Optional[str] = None;

    @property
    def value(self):
        return self.slider_value * 10**self.exponent

    @value.setter
    def value(self, new_val):
        self.exponent = int(np.log10(new_val))
        self.slider_value = int(new_val/10**self.exponent)

    @staticmethod
    def from_any(obj: Any) -> "SliderSettings":
        """
        Erstellt ein SliderSettings-Objekt aus:
        - einem SliderSettings-Objekt (gibt es direkt zurück)
        - einem dict mit keys 'slider_value' und 'exponent' oder 'exp'
        """
        if isinstance(obj, SliderSettings):
            return obj
        if isinstance(obj, dict):
            # handle both possible keys
            val = obj.get("value", None)
            if val:
                exp = int(np.log10(val))
                slider_val = int(val/10**exp)
            else:
                exp = obj.get("exponent", obj.get("exp", 0))
                slider_val = obj.get("slider_value", 1.0)
            return SliderSettings(
                name = obj.get("name", None),
                exponent = exp,
                slider_value = slider_val,
            )
        if isinstance(obj, InitialParameter):
            exp = int(np.log10(obj.value))
            val = int(obj.value / 10**exp)
            return SliderSettings(
                name = obj.name,
                exponent=exp,
                slider_value=val,
            )
        if isinstance(obj, (int, float, np.integer, np.floating)):
            x = SliderSettings(0, 0)
            x.value = float(obj)
            return x;
        raise TypeError(f"Cannot convert {obj} to SliderSettings")

# ==================================================

def _save_params(file: Path, params: dict[str, SliderSettings]):
    """Speichert SliderSettings als JSON"""
    serializable = {}
    for k, v in params.items():
        if isinstance(v, SliderSettings):
            serializable[k] = asdict(v)
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            raise TypeError(f"Unexpected value type for {k}: {type(v)}")
    file.write_text(json.dumps(serializable, indent=2))

def _load_params(file: Path, default: dict[str, SliderSettings]) -> dict[str, SliderSettings]:
    """Lädt JSON und wandelt dicts in SliderSettings um"""
    if not file.exists():
        return default

    raw = json.loads(file.read_text())
    loaded = {}
    for k, v in raw.items():
        loaded[k] = SliderSettings.from_any(v)
    # default values als Fallback ergänzen
    for k, v in default.items():
        if k not in loaded:
            loaded[k] = v
    return loaded

def _extract_default_values(
    x_data,
    y_data,
    model: Union[Callable, FitModel],
    user_specified: Optional[Union[List[float], dict[str, float]]] = None
) -> dict[str, SliderSettings]:
    params = list(inspect.signature(model).parameters.keys())[1:]

    # ensure all params have some kind of default
    default = { 
        p: SliderSettings(
            name = p,
            slider_value=1,
            exponent=0
        ) for p in params
    }

    from_guess = {} 
    if isinstance(model, FitModel):
        initial_guess = model.get_initial_guess(x_data, y_data);
        from_guess = { 
            p: SliderSettings.from_any(initial_guess[i]) for i, p in enumerate(params)
        };
        default = {**default, **from_guess}

    if not user_specified:
        return default

    if isinstance(user_specified, List):
        from_values = {
            p: SliderSettings.from_any(user_specified[i]) for i, p in enumerate(params)
        }

    if isinstance(default_values, List):
        from_values = {p : default_values[i] for i, p in enumerate(params)}
        return {**std_default, **from_guess}
        default = {**default, **from_values}
    elif isinstance(default_values, dict):
        default = {**default, **default_values}
    else:
        raise ValueError("")

def find_init_params(
    x_data,
    y_data,
    model: Union[Callable, FitModel],
    cachePath="fitcache.json",
    default_values: Optional[Union[List[float], dict[str, float]]] = None
) -> dict[str, float]:
    filepath = ensure_extension(cachePath, ".json")
    CACHE = Path(filepath);

    fig, ax = plt.subplots()
    ax.set_title("Find Parameters")
    ax.scatter(x_data, y_data)

    params = list(inspect.signature(model).parameters.keys())[1:]
    
    default = _extract_default_values(x_data, y_data, model, default_values)
    slider_settings: dict[str, SliderSettings] = _load_params(CACHE, default=default) 

    sliderheight = 0.05
    offset = sliderheight * len(params) + 2 * sliderheight
    fig.subplots_adjust(bottom=offset)

    x_plot = np.linspace(x_data.min(), x_data.max(), 200)
    line, = ax.plot(x_plot, model(x_plot, *[slider_settings[p].value for p in params]), color='red')

    # since garbage collector would remove the references we store them
    sliders = {}
    btns_plus = {}
    btns_minus = {}
    exp_texts = {}
    textboxes = {}

    last_update_time = 0

    def update_graph(param):
        # global last_update_time
        # # Debounce: mindestens 0.05s Abstand zwischen Redraws
        # now = time.time()
        # if now - last_update_time < 0.05:
        #     return
        # last_update_time = now

        y_plot = model(x_plot, *[slider_settings[p].value for p in params])
        line.set_ydata(y_plot)

        paramval = slider_settings[p].value

        # update texts
        if p in textboxes:
            textboxes[p].set_text(paramval)

        # Achsen automatisch anpassen
        ymin = min(y_data.min(), y_plot.min())
        ymax = max(y_data.max(), y_plot.max())
        dy = ymax - ymin or 1.0
        pad = 0.05 * dy
        ax.set_ylim(ymin - pad, ymax + pad)


        # Canvas aktualisieren
        fig.canvas.draw_idle()

        _save_params(CACHE, param_values)

    for i, p in enumerate(params):
        # UI-Positionen
        y_pos = offset - (i+2) * sliderheight
        slider_ax = plt.axes([0.15, y_pos, 0.35, sliderheight])
        btn_minus_ax = plt.axes([0.75, y_pos, 0.05, 0.8 * sliderheight])
        btn_plus_ax  = plt.axes([0.825, y_pos, 0.05, 0.8 * sliderheight])
        # text_ax = plt.axes([0.505, y_pos + 0.005, 0.1, 0.8*sliderheight])

         # Label for exponent
        exp_ax = plt.axes([0.60, y_pos + 0.005, 0.04, 0.8*sliderheight])
        exp_text = exp_ax.text(0.5, 0.5, fr"$\times 10^{{{slider_settings[p].exponent}}}$", ha='left', va='center')
        exp_ax.axis('off')  # hide axes
        exp_texts[p] = exp_text


        # UI-Elemente
        slider = Slider(slider_ax, p, 0.1, 12.0, valinit=slider_settings[p].slider_value)
        btn_minus = Button(btn_minus_ax, '-')
        btn_plus  = Button(btn_plus_ax, '+')
        # text_box = TextBox(text_ax, "", color="white", hovercolor="lightgrey")
        # text_box.text_disp.set_color("black")

        # since garbage collector would remove; I NEED these references:
        sliders[p] = slider
        btns_plus[p] = btn_plus
        btns_minus[p] = btn_minus
        # textboxes[p] = text_box

        # def make_update_textbox(param):
        #     def update(text):
        #         print("Du hast eingegeben:", text, "for", param)
        #
        #     return update;
        #
        # text_box.on_submit(make_update_textbox(p))

        # Slider-Callback
        def make_update(param):
            def update(val, param=param):  # <-- fixiert param
                slider_settings[param].slider_value = val
                update_graph(param)
            return update
        slider.on_changed(make_update(p))

        # Buttons-Callback
        def make_exponent_buttons(param, slider, btn_plus, btn_minus):
            def increase(event, param=param, slider=slider):
                slider_settings[param].exponent += 1
                slider.set_val(slider_settings[param].slider_value / 10)
                exp_texts[param].set_text(fr"$\times 10^{{{slider_settings[p].exponent}}}$")
                update_graph(param)

            def decrease(event, param=param, slider=slider):
                slider_settings[param].exponent -= 1
                slider.set_val(slider_settings[param].slider_value * 10)
                exp_texts[param].set_text(fr"$\times 10^{{{slider_settings[p].exponent}}}$")
                update_graph(param)

            btn_plus.on_clicked(increase)
            btn_minus.on_clicked(decrease)

        make_exponent_buttons(p, slider, btn_plus, btn_minus)

    plt.show()
    _save_params(CACHE, slider_settings)

    initial_params = {}
    for k, v in slider_settings.items():
        initial_params[k] = v.value;

    return initial_params
