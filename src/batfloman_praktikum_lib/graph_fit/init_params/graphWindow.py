from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from .order_init_params import order_initial_params

class GraphWindow(QWidget):
    param_win = None

    def __init__(self,
        x_data, 
        y_data, 
        model, 
        params: dict[str, float]
    ):
        super().__init__()
        self.setWindowTitle("Model Visualization")

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Store data
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.model = model
        self.params = params

        # Scatter plot for data
        self.scatter = pg.ScatterPlotItem(
            x=self.x_data,
            y=self.y_data,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 150),
            size=6
        )
        self.plot_widget.addItem(self.scatter)

        # Line plot for model
        self.x_line = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        self.line_plot = self.plot_widget.plot(
            self.x_line,
            self.model(self.x_line, *list(params.values())),
            pen=pg.mkPen('g', width=2)
        )

    def update_params(self, new_params: dict[str, float]):
        """Update the line plot with new parameter values"""
        self.params = new_params
        y_line = self.model(self.x_line, *order_initial_params(self.model, new_params))
        self.line_plot.setData(self.x_line, y_line)

    def keyPressEvent(self, a0) -> None:  # use base name 'a0' to satisfy checker
        if a0.key() == Qt.Key.Key_Q:
            self.close()
            if self.param_win is not None:
                self.param_win.close()
        else:
            super().keyPressEvent(a0)
