from dataclasses import dataclass, asdict
from typing import Optional, Any, Union
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from pathlib import Path
import json
import inspect

from ..tables.validation import ensure_extension

@dataclass
class SliderSettings:
    slider_value: float
    exponent: int = 0

    def get_value(self) -> float:
        return self.slider_value * 10 ** self.exponent

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
            slider_val = obj.get("slider_value", 1.0)
            exponent = obj.get("exponent", obj.get("exp", 0))
            return SliderSettings(slider_val, exponent)
        raise TypeError(f"Cannot convert {obj} to SliderSettings")

def _save_params(file: Path, params: dict[str, Union[SliderSettings, dict]]):
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

def find_init_params(
    x_data,
    y_data,
    model,
    cachePath="fitcache.json",
    index: int | None = None
) -> dict[str, SliderSettings]:
    filepath = ensure_extension(cachePath, ".json")
    CACHE = Path(filepath);

    fig, ax = plt.subplots()
    ax.set_title("Find Parameters")
    ax.scatter(x_data, y_data)

    params = list(inspect.signature(model).parameters.keys())[1:]
    param_values: dict[str, SliderSettings] = _load_params(CACHE, default={ 
        p: SliderSettings(slider_value=1, exponent=0) for p in params 
    })

    sliderheight = 0.05
    offset = sliderheight * len(params) + 2 * sliderheight
    fig.subplots_adjust(bottom=offset)

    x_plot = np.linspace(x_data.min(), x_data.max(), 200)
    line, = ax.plot(x_plot, model(x_plot, *[param_values[p].get_value() for p in params]), color='red')

    def update_graph():
        y_plot = model(x_plot, *[param_values[p].get_value() for p in params])
        line.set_ydata(y_plot)

        # Achsen automatisch anpassen
        ymin = min(y_data.min(), y_plot.min())
        ymax = max(y_data.max(), y_plot.max())
        dy = ymax - ymin or 1.0
        pad = 0.05 * dy
        ax.set_ylim(ymin - pad, ymax + pad)

        # Canvas aktualisieren
        fig.canvas.draw_idle()

        _save_params(CACHE, param_values)

    sliders = {}
    btns_plus = {}
    btns_minus = {}
    exp_texts = {}
    for i, p in enumerate(params):
        # UI-Positionen
        y_pos = offset - (i+2) * sliderheight
        slider_ax = plt.axes([0.15, y_pos, 0.35, sliderheight])
        btn_minus_ax = plt.axes([0.75, y_pos, 0.05, 0.8 * sliderheight])
        btn_plus_ax  = plt.axes([0.825, y_pos, 0.05, 0.8 * sliderheight])

         # Label for exponent
        exp_ax = plt.axes([0.60, y_pos + 0.005, 0.04, 0.8*sliderheight])
        exp_text = exp_ax.text(0.5, 0.5, fr"$\times 10^{{{param_values[p].exponent}}}$", ha='left', va='center')
        exp_ax.axis('off')  # hide axes
        exp_texts[p] = exp_text

        # UI-Elemente
        slider = Slider(slider_ax, p, 0.1, 12.0, valinit=param_values[p].slider_value)
        btn_minus = Button(btn_minus_ax, '-')
        btn_plus  = Button(btn_plus_ax, '+')

        # since garbage collector would remove; I NEED these references:
        sliders[p] = slider
        btns_plus[p] = btn_plus
        btns_minus[p] = btn_minus

        # Slider-Callback
        def make_update(param):
            def update(val, param=param):  # <-- fixiert param
                param_values[param].slider_value = val
                update_graph()
            return update
        slider.on_changed(make_update(p))

        # Buttons-Callback
        def make_exponent_buttons(param, slider, btn_plus, btn_minus):
            def increase(event, param=param, slider=slider):
                param_values[param].exponent += 1
                slider.set_val(param_values[param].slider_value / 10)
                exp_texts[param].set_text(fr"$\times 10^{{{param_values[p].exponent}}}$")
                update_graph()

            def decrease(event, param=param, slider=slider):
                param_values[param].exponent -= 1
                slider.set_val(param_values[param].slider_value * 10)
                exp_texts[param].set_text(fr"$\times 10^{{{param_values[p].exponent}}}$")
                update_graph()

            btn_plus.on_clicked(increase)
            btn_minus.on_clicked(decrease)

        make_exponent_buttons(p, slider, btn_plus, btn_minus)

    plt.show()
    _save_params(CACHE, param_values)
    return param_values
