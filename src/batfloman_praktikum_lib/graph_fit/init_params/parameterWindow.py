from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from .parameterSlider import ParameterSlider

class ParameterWindow(QWidget):
    graph_win = None

    def __init__(self, params: dict[str, float | ParameterSlider], update_callback):
        super().__init__()

        self.setWindowTitle("Manual Initial Parameters")
        layout = QVBoxLayout(self)
        self.sliders = {}

        for name, val in params.items():
            if isinstance(val, ParameterSlider):
                s = val
                s.update_callback = update_callback
            else:  # assume float, create new slider
                s = ParameterSlider(
                    name=name,
                    initial_value=val,
                    vmin=0.5 * val,
                    vmax=1.5 * val,
                    update_callback=update_callback
                )

            layout.addWidget(s)
            self.sliders[name] = s

    def get_params(self):
        return {k: s.get_value() for k, s in self.sliders.items()}

    def keyPressEvent(self, a0) -> None:  # use base name 'a0' to satisfy checker
        if a0.key() == Qt.Key.Key_Q:
            self.close()
            if self.graph_win is not None:
                self.graph_win.close()
        else:
            super().keyPressEvent(a0)
