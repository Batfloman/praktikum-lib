from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np

from batfloman_praktikum_lib.graph_fit.init_params._helper import smart_format

from .order_init_params import order_initial_params

class GraphWindow(QWidget):
    param_win = None
    line_sample_count = 1000

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

        # Label to show mouse position
        self.coord_label = QLabel("x=0, y=0")
        layout.addWidget(self.coord_label)

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
        self.x_line = self._get_data_x_line()
        self.line_plot = self.plot_widget.plot(
            self.x_line,
            self.model(self.x_line, *list(params.values())),
            pen=pg.mkPen('g', width=2)
        )

        # Connect mouse movement to update coordinates
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        QTimer.singleShot(0, self._initialize_visible_curve)

    def mouse_moved(self, pos):
        """Update coordinate label with current mouse position"""
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self.coord_label.setText(f"x={smart_format(mouse_point.x())}, y={smart_format(mouse_point.y())}")

    def _get_data_x_line(self):
        return np.linspace(float(np.min(self.x_data)), float(np.max(self.x_data)), self.line_sample_count)

    def _get_visible_x_line(self):
        x_min, x_max = self.plot_widget.plotItem.vb.viewRange()[0]
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            return self._get_data_x_line()
        return np.linspace(x_min, x_max, self.line_sample_count)

    def _initialize_visible_curve(self):
        self.update_visible_model_curve()
        self.plot_widget.enableAutoRange(axis='x', enable=False)
        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self.update_visible_model_curve)

    def update_visible_model_curve(self, *_args):
        self.x_line = self._get_visible_x_line()
        y_line = self.model(self.x_line, *order_initial_params(self.model, self.params))
        self.line_plot.setData(self.x_line, y_line)

    def update_params(self, new_params: dict[str, float]):
        """Update the line plot with new parameter values"""
        self.params = new_params
        self.update_visible_model_curve()

    def keyPressEvent(self, a0) -> None:  # use base name 'a0' to satisfy checker
        if a0.key() == Qt.Key.Key_Q:
            self.close()
            if self.param_win is not None:
                self.param_win.close()
        else:
            super().keyPressEvent(a0)
