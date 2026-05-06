from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np

from batfloman_praktikum_lib.graph_fit.init_params._helper import smart_format

from .order_init_params import order_initial_params
from .render_parts import RenderPart

class GraphWindow(QWidget):
    param_win = None
    line_sample_count = 1000

    def __init__(self,
        x_data, 
        y_data, 
        model, 
        params: dict[str, float],
        render_parts: Optional[list[RenderPart]] = None,
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
        self.data_x_min = float(np.min(self.x_data))
        self.data_x_max = float(np.max(self.x_data))
        self.model = model
        self.params = params
        self.render_parts = render_parts or []
        self.render_part_plots: dict[str, pg.PlotDataItem] = {}
        self.render_part_visibility: dict[str, bool] = {}
        self.plot_item_to_render_part_key: dict[pg.PlotDataItem, str] = {}
        self._update_scheduled = False

        self.legend = self.plot_widget.addLegend()

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
            self.model(self.x_line, *order_initial_params(self.model, self.params)),
            pen=pg.mkPen('g', width=2),
            name="Model",
        )

        render_part_colors = ["b", "m", "c", "y", "w"]
        for idx, part in enumerate(self.render_parts):
            pen = pg.mkPen(part.color or render_part_colors[idx % len(render_part_colors)], width=2, style=Qt.PenStyle.DashLine)
            plot = self.plot_widget.plot([], [], pen=pen, name=part.label)
            plot.setVisible(part.visible_by_default)
            self.render_part_plots[part.key] = plot
            self.render_part_visibility[part.key] = part.visible_by_default
            self.plot_item_to_render_part_key[plot] = part.key
            plot.sigPlotChanged.connect(
                lambda *args, key=part.key: self._handle_render_part_plot_changed(key)
            )

        # Connect mouse movement to update coordinates
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        self.legend.sigSampleClicked.connect(self._handle_legend_sample_clicked)
        QTimer.singleShot(0, self._initialize_visible_curve)

    def mouse_moved(self, pos):
        """Update coordinate label with current mouse position"""
        vb = self.plot_widget.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self.coord_label.setText(f"x={smart_format(mouse_point.x())}, y={smart_format(mouse_point.y())}")

    def _get_data_x_line(self):
        return np.linspace(self.data_x_min, self.data_x_max, self.line_sample_count)

    def _get_visible_x_line(self):
        x_min, x_max = self.plot_widget.plotItem.vb.viewRange()[0]
        if (
            not np.isfinite(x_min)
            or not np.isfinite(x_max)
            or abs(x_max - x_min) < 1e-12
        ):
            return self._get_data_x_line()
        return np.linspace(x_min, x_max, self.line_sample_count)

    def _initialize_visible_curve(self):
        self.plot_widget.setXRange(self.data_x_min, self.data_x_max, padding=0.02)
        self.update_visible_model_curve()
        self.plot_widget.plotItem.sigRangeChanged.connect(self.schedule_visible_model_curve_update)
        self.plot_widget.plotItem.sigRangeChangedManually.connect(self.schedule_visible_model_curve_update)
        self.plot_widget.plotItem.vb.sigStateChanged.connect(self.schedule_visible_model_curve_update)
        self.plot_widget.plotItem.vb.sigTransformChanged.connect(self.schedule_visible_model_curve_update)
        self.plot_widget.sigTransformChanged.connect(self.schedule_visible_model_curve_update)

    def schedule_visible_model_curve_update(self, *_args):
        if self._update_scheduled:
            return
        self._update_scheduled = True
        QTimer.singleShot(0, self._flush_visible_model_curve_update)

    def _flush_visible_model_curve_update(self):
        self._update_scheduled = False
        self.update_visible_model_curve()

    def _handle_legend_sample_clicked(self, item):
        key = self.plot_item_to_render_part_key.get(item)
        if key is None:
            return
        self._sync_render_part_visibility(key, item.isVisible(), schedule_update=True)

    def _handle_render_part_plot_changed(self, key: str):
        plot = self.render_part_plots[key]
        visible = plot.isVisible()
        if self.render_part_visibility.get(key) == visible:
            return
        self._sync_render_part_visibility(key, visible, schedule_update=True)

    def _sync_render_part_visibility(self, key: str, visible: bool, *, schedule_update: bool):
        self.render_part_visibility[key] = visible

        plot = self.render_part_plots[key]
        if plot.isVisible() != visible:
            plot.setVisible(visible)
        if not visible:
            plot.setData([], [])

        if self.param_win is not None:
            self.param_win.set_render_part_visibility(key, visible)

        if schedule_update:
            self.schedule_visible_model_curve_update()

    def update_visible_model_curve(self, *_args):
        self.x_line = self._get_visible_x_line()
        y_line = self.model(self.x_line, *order_initial_params(self.model, self.params))
        self.line_plot.setData(self.x_line, y_line)
        for part in self.render_parts:
            plot = self.render_part_plots[part.key]
            if not plot.isVisible():
                continue
            plot.setData(self.x_line, part.evaluator(self.x_line, self.params))

    def update_params(self, new_params: dict[str, float]):
        """Update the line plot with new parameter values"""
        self.params = new_params
        self.schedule_visible_model_curve_update()

    def set_render_part_visibility(self, key: str, visible: bool):
        self._sync_render_part_visibility(key, visible, schedule_update=True)

    def keyPressEvent(self, a0) -> None:  # use base name 'a0' to satisfy checker
        if a0.key() == Qt.Key.Key_Q:
            self.close()
            if self.param_win is not None:
                self.param_win.close()
        else:
            super().keyPressEvent(a0)
