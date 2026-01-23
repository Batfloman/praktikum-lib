from typing import Optional
from matplotlib.widgets import Slider, Button
from pathlib import Path
import json

class ParameterSlider:
    slider: Slider;

    slider_value: float

    center: float
    vmin: float
    vmax: float

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
        self.vmin = min or 0.5 * initial_value
        self.vmax = max or 1.5 * initial_value

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
        create_slider(ax=slider_ax, label=param, valmin=self.vmin, valmax=self.vmax, valinit=initial_value)

        def get_delta_text():
           # return f"Δ = {self.max - self.min:.2e}"
            return fr"Range $({self.center:.3g} \pm {abs(self.vmax - self.vmin)/2:.2g})$"

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
            create_slider(ax=ax, label=label, valmin=self.vmin, valmax=self.vmax, valinit=val)
            self.slider.on_changed(slider_callback)  # reattach callback

        def btn_center_callback(event):
            offset = self.slider_value - self.center
            self.vmin += offset
            self.vmax += offset
            self.center += offset

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b2.on_clicked(btn_center_callback);

        def btn_minimize_callback(event):
            delta = abs(self.vmin - self.vmax)
            new_delta = 0.5 * delta;

            self.vmin = self.center - 0.5 * new_delta
            self.vmax = self.center + 0.5 * new_delta

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b1.on_clicked(btn_minimize_callback)

        def btn_expand_callback(event):
            delta = abs(self.vmin - self.vmax)
            new_delta = 2 * delta;

            self.vmin = self.center - 0.5 * new_delta
            self.vmax = self.center + 0.5 * new_delta

            self.text_delta.set_text(get_delta_text())
            recreate_slider()

        self.b3.on_clicked(btn_expand_callback)

    def get_value(self):
        return self.slider_value

    def to_dict(self):
        return {
            "min": self.vmin,
            "max": self.vmax,
            "center": self.center,
            "slider_value": self.slider_value,
        }

def save_slider_settings(file: Path, params: dict[str, ParameterSlider]):
    serializable = {}
    for k, v in params.items():
        serializable[k] = v.to_dict()
    file.write_text(json.dumps(serializable, indent=2))

def load_slider_settings(file: Path) -> dict[str, dict]:
    d = {};
    if not file.exists():
        return d;

    return json.loads(file.read_text())
