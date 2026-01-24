from typing import Callable, Optional
from PIL.Image import init
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import json

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ._widget_helper import mult_slider_range, recenter_slider

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

    def __init__(self,
        param: str,
        total_ax, 
        initial_value: float,
        update_callback: Optional[Callable],
        min: Optional[float] = None,
        max: Optional[float] = None,
        center: Optional[float] = None,
    ):
        center = center or initial_value
        vmin = min or 0.5 * center
        vmax = max or 1.5 * center

        # ==================================================
        # style 

        total_ax.set_xticks([])
        total_ax.set_yticks([])
        total_ax.set_frame_on(False)

        # --------------------
        # slider

        slider_ax = inset_axes(
            total_ax, 
            width="60%", 
            height="50%", 
            loc='center left'
        )
        slider = Slider(slider_ax, label=param, valmin=vmin, valmax=vmax, valinit=initial_value)

        slider.label.set_x(1.02)
        slider.label.set_horizontalalignment("left")
        slider.valtext.set_visible(False)

        self.slider = slider

        # --------------------
        # range text

        self.text_delta = total_ax.text(
            0.31, 0.9, "",
            ha="center",
            va="center",
            transform=total_ax.transAxes
        )

        # --------------------
        # textbox

        textbox_ax = inset_axes(
            slider_ax,
            width="16%",      # Breite der TextBox
            height="100%",    # volle Höhe der Slider-Achse
            loc="center left", 
            bbox_to_anchor=(-0.19, 0, 1, 1),  # x-Offset links, feintunen je nach Abstand
            bbox_transform=slider_ax.transAxes,
        )
        textbox = TextBox(textbox_ax, label="", initial=f"{_smart_format(initial_value)}")

        self.text_box = textbox

        # --------------------
        # buttons

        def create_button(total_ax, pos, label) -> Button:
            btn_ax = inset_axes(
                total_ax,
                bbox_to_anchor=(pos, 0, 0.1, 1),
                bbox_transform=total_ax.transAxes,
                width="80%", height="80%",
                loc="center"
            )
            return Button(btn_ax, label)

        self.b_shrink = create_button(total_ax, pos=0.75, label='–')
        self.b_center = create_button(total_ax, pos=0.85, label='●')
        self.b_expand = create_button(total_ax, pos=0.95, label='+')

        # ==================================================
        # events

        # --------------------
        # slider-range visual indicator

        def update_delta_text():
            center = (self.slider.valmin + self.slider.valmax) / 2
            delta = abs(self.slider.valmax - self.slider.valmin) / 2
            text = fr"Range $({_smart_format(center)} \pm {_smart_format(delta)})$"

            self.text_delta.set_text(text)

        update_delta_text() # call once at the start

        # --------------------
        # param_val changers = slider, textbox

        self._syncing = False

        def on_slider_change(val):
            if self._syncing:
                return
            self._syncing = True
            self.text_box.set_val(_smart_format(val))
            self._syncing = False

            if update_callback:
                update_callback()

        self.slider.on_changed(on_slider_change)

        def on_textbox_submit(text):
            if self._syncing:
                return
            try:
                val = float(text)
            except ValueError:
                return

            self._syncing = True
            self.slider.set_val(val)
            self._syncing = False
            recenter_slider(self.slider)
            update_delta_text()
            self.text_box.stop_typing()

            if update_callback:
                update_callback()

        self.text_box.on_submit(on_textbox_submit)

        # --------------------
        # buttons

        def shrink(event):
            mult_slider_range(self.slider, 0.5)
            update_delta_text()

        def expand(event):
            mult_slider_range(self.slider, 2)
            update_delta_text()

        def recenter(event):
            recenter_slider(self.slider)
            update_delta_text()

        self.b_shrink.on_clicked(shrink)
        self.b_expand.on_clicked(expand)
        self.b_center.on_clicked(recenter)

    # ==================================================

    def to_dict(self):
        return {
            "min": self.slider.valmin,
            "max": self.slider.valmax,
            "center": (self.slider.valmin + self.slider.valmax) / 2,
            "slider_value": self.slider.val,
        }

    @property
    def slider_value(self):
        return self.slider.val

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
