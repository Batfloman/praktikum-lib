from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from batfloman_praktikum_lib.graph_fit.init_params._helper import smart_format

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
        self.x_line = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        self.line_plot = self.plot_widget.plot(
            self.x_line,
            self.model(self.x_line, *list(params.values())),
            pen=pg.mkPen('g', width=2)
        )

        # Connect mouse movement to update coordinates
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)

    def mouse_moved(self, pos):
        """Update coordinate label with current mouse position"""
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self.coord_label.setText(f"x={smart_format(mouse_point.x())}, y={smart_format(mouse_point.y())}")

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
