from typing import Optional
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import json

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def _smart_format(val: float, sig: int = 3) -> str:
    if val == 0:
        return "0"
    if abs(val) >= 1e5 or abs(val) < 1e-2:
        return f"{val:.{sig}e}"
    if abs(val) >= 1e3:
        return f"{val:.0f}"
    else:
        return f"{val:.{sig}g}"

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
        self.slider_value = initial_value

        total_ax.set_xticks([])
        total_ax.set_yticks([])
        total_ax.set_frame_on(False)

        slider_ax = inset_axes(total_ax, width="60%", height="50%", loc='center left')
        self._create_slider(slider_ax, update_callback=update_callback)

        self.text_delta = total_ax.text(0.31, 0.9, self._get_delta_text(), ha="center", va="center", transform=total_ax.transAxes)

        self.b_shrink = self._create_button(total_ax, pos=0.75, label='–', callback=self._shrink, update_callback=update_callback)
        self.b_center = self._create_button(total_ax, pos=0.85, label='●', callback=self._recenter, update_callback=update_callback)
        self.b_expand = self._create_button(total_ax, pos=0.95, label='+', callback=self._expand, update_callback=update_callback)

    def get_value(self):
        return self.slider_value

    def to_dict(self):
        return {
            "min": self.vmin,
            "max": self.vmax,
            "center": self.center,
            "slider_value": self.slider_value,
        }

    def _get_delta_text(self):
        # return f"Δ = {self.max - self.min:.2e}"
        vrange = abs(self.vmax - self.vmin) / 2
        return fr"Range $({self.center:.3g} \pm {_smart_format(vrange)})$"

    def _create_slider(self, ax, update_callback):
        self.slider = Slider(ax, label=self.param, valmin=self.vmin, valmax=self.vmax, valinit=self.slider_value)
        self.slider.label.set_x(1.02)
        self.slider.label.set_horizontalalignment("left")
        # self.slider.valtext.set_x(-0.01)
        # self.slider.valtext.set_horizontalalignment("right")
        # alte valtext ausblenden
        self.slider.valtext.set_visible(False)

        self._create_textbox(ax, self.slider_value, update_callback)

        def callback(val):
            self.slider_value = val;
            update_callback()

        self.slider.on_changed(callback)

    def _create_textbox(self, slider_ax, valinit, update_callback):
        # TextBox links neben Slider
        textbox_ax = inset_axes(
            slider_ax,
            width="16%",      # Breite der TextBox
            height="100%",    # volle Höhe der Slider-Achse
            loc="center left", 
            bbox_to_anchor=(-0.19, 0, 1, 1),  # x-Offset links, feintunen je nach Abstand
            bbox_transform=slider_ax.transAxes,
        )
        self.text_box_ax = textbox_ax
        self.text_box = TextBox(textbox_ax, label="", initial=f"{valinit:.3g}")

        self._updating_from_slider = False

        def slider_callback(val):
            self.slider_value = val
            self._updating_from_slider = True
            self.text_box.set_val(f"{val:.3g}")
            self._updating_from_slider = False
        self.slider.on_changed(slider_callback)

        def textbox_submit(text):
            if self._updating_from_slider:
                return  # verhindern, dass der Slider-Update das Submit erneut auslöst
            try:
                self.slider_value = float(text)
                self._recenter(update_callback)
            except ValueError:
                pass

        self.text_box.on_submit(textbox_submit)

    def _recreate_slider(self, update_callback):
        ax = self.slider.ax
        ax.clear()

        if hasattr(self, "text_box_ax"):
            self.text_box_ax.remove()
            del self.text_box_ax
            del self.text_box

        self._create_slider(ax=ax, update_callback=update_callback)

    def _create_button(self, total_ax, pos, label, callback, update_callback):
        btn_ax = inset_axes(
            total_ax,
            bbox_to_anchor=(pos, 0, 0.1, 1),
            bbox_transform=total_ax.transAxes,
            width="80%", height="80%",
            loc="center"
        )

        btn = Button(btn_ax, label)
        btn.on_clicked(lambda event, cb=callback, update_callback=update_callback: cb(update_callback))

        return btn 

    # Button-Logik direkt hier
    def _shrink(self, update_callback):
        delta = abs(self.vmax - self.vmin)
        new_delta = 0.5 * delta
        self.vmin = self.center - 0.5 * new_delta
        self.vmax = self.center + 0.5 * new_delta
        self.text_delta.set_text(self._get_delta_text())
        self.slider.valmin = self.vmin
        self.slider.valmax = self.vmax

        self._recreate_slider(update_callback)
        update_callback()

    def _expand(self, update_callback):
        delta = abs(self.vmax - self.vmin)
        new_delta = 2 * delta
        self.vmin = self.center - 0.5 * new_delta
        self.vmax = self.center + 0.5 * new_delta
        self.text_delta.set_text(self._get_delta_text())
        self.slider.valmin = self.vmin
        self.slider.valmax = self.vmax

        self._recreate_slider(update_callback)
        update_callback()

    def _recenter(self, update_callback):
        offset = self.slider_value - self.center
        self.center += offset
        self.vmin += offset
        self.vmax += offset
        self.text_delta.set_text(self._get_delta_text())
        self.slider.valmin = self.vmin
        self.slider.valmax = self.vmax

        self._recreate_slider(update_callback)
        update_callback()


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
