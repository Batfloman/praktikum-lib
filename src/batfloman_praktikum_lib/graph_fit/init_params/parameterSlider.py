from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QDoubleSpinBox, QPushButton, QLabel
)
from PyQt6.QtCore import Qt

from batfloman_praktikum_lib.graph_fit.init_params._helper import smart_format

class ParameterSlider(QWidget):
    def __init__(self,
        name: str,
        initial_value: float,
        *,
        center: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        update_callback = None,
    ):
        super().__init__()

        self.name = name
        self.update_callback = update_callback

        self.slider_value = initial_value
        self.center = center or initial_value
        self.vmin = vmin or (self.center * 0.5 if self.center != 0 else -1.0)
        self.vmax = vmax or (self.center * 1.5 if self.center != 0 else 1.0)

        self._syncing = False

        # ---------------- layout ----------------
        layout = QHBoxLayout(self)

        self.label = QLabel(name)
        layout.addWidget(self.label)

        slider_container = QVBoxLayout()
        slider_container.setContentsMargins(0, 0, 0, 0)  # no extra spacing

        # range label on top
        self.range_label = QLabel(self._format_range_text())
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        slider_container.addWidget(self.range_label)

        # the slider below
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        slider_container.addWidget(self.slider)

        # add this vertical container to your main horizontal layout
        layout.addLayout(slider_container, stretch=1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(6)
        self.spin.setRange(-1e12, 1e12)
        layout.addWidget(self.spin)

        self.btn_shrink = QPushButton("–")
        self.btn_center = QPushButton("●")
        self.btn_expand = QPushButton("+")

        layout.addWidget(self.btn_shrink)
        layout.addWidget(self.btn_center)
        layout.addWidget(self.btn_expand)

        # ---------------- init ----------------
        self.set_value(initial_value)

        # ---------------- signals ----------------
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

        self.btn_center.clicked.connect(self.recenter)
        self.btn_shrink.clicked.connect(lambda: self.scale_range(0.5))
        self.btn_expand.clicked.connect(lambda: self.scale_range(2.0))

    # ==========================================
    # mapping

    def _slider_to_value(self, i: int) -> float:
        return self.vmin + (self.vmax - self.vmin) * i / 1000

    def _value_to_slider(self, v: float) -> int:
        return int(1000 * (v - self.vmin) / (self.vmax - self.vmin))

    # ==========================================
    # sync

    def _on_slider(self, i):
        if self._syncing:
            return
        self._syncing = True
        v = self._slider_to_value(i)
        self.spin.setValue(v)
        self.slider_value = v
        self._syncing = False
        self._update_range_label()
        self._emit_update()

    def _on_spin(self, v):
        if self._syncing:
            return
        self._syncing = True
        self.slider_value = v
        self.slider.setValue(self._value_to_slider(v))
        self._syncing = False
        self.recenter()
        self._update_range_label()
        self._emit_update()

    # ==========================================
    # operations

    def _format_range_text(self):
        delta = abs(self.vmax - self.vmin) / 2
        return f"Range: ({smart_format(self.center)} ± {smart_format(delta)})"

    def _update_range_label(self):
        self.range_label.setText(self._format_range_text())

    def recenter(self):
        delta = self.slider_value - self.center
        self.vmin += delta
        self.vmax += delta
        self.center = (self.vmin + self.vmax) / 2  # recompute center properly
        self.slider.setValue(self._value_to_slider(self.slider_value))
        self._update_range_label()

    def scale_range(self, factor: float):
        half = (self.vmax - self.vmin) / 2 * factor
        self.vmin = self.center - half
        self.vmax = self.center + half
        self.slider.setValue(self._value_to_slider(self.slider_value))
        self._update_range_label()

    # ==========================================

    def set_value(self, v: float):
        self.slider_value = v
        self.spin.setValue(v)
        self.slider.setValue(self._value_to_slider(v))

    def get_value(self) -> float:
        return self.slider_value

    def to_dict(self):
        return {
            "min": self.vmin,
            "max": self.vmax,
            "center": self.center,
            "slider_value": self.slider_value,
        }

    def _emit_update(self):
        if self.update_callback:
            self.update_callback()

    @classmethod
    def from_cache(cls, name: str, cache: dict, default_value: float, update_callback=None):
        """
        Create a ParameterSlider using cached settings if available.
        cache: dict from load_slider_settings, keyed by parameter name
        default_value: fallback if no cached value
        """
        settings = cache.get(name, {})
        initial_value = settings.get("slider_value", default_value)
        center = settings.get("center", initial_value)
        vmin = settings.get("min", None)
        vmax = settings.get("max", None)
        return cls(
            name=name,
            initial_value=initial_value,
            center=center,
            vmin=vmin,
            vmax=vmax,
            update_callback=update_callback
        )


# ==================================================
# import/exporting values
# ==================================================

from pathlib import Path
import json

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
