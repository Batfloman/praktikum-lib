from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np

from batfloman_praktikum_lib.graph_fit.init_params._helper import smart_format
from batfloman_praktikum_lib.graph_fit.helper import evaluate_model

from .order_init_params import order_initial_params
from .render_parts import DEFAULT_RENDER_PART_COLORS, RenderPart

class GraphWindow(QWidget):
    param_win = None
    fit_selection_win = None
    line_sample_count = 1000
    on_close_callback = None

    def __init__(self,
        x_data, 
        y_data, 
        model, 
        params: dict[str, float],
        render_parts: Optional[list[RenderPart]] = None,
        interval_indices: tuple[int, int] | None = None,
        excluded_indices: tuple[int, ...] = (),
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
        self.point_count = len(self.x_data)
        self.model = model
        self.params = params
        self.render_parts = render_parts or []
        self.render_part_plots: dict[str, pg.PlotDataItem] = {}
        self.render_part_visibility: dict[str, bool] = {}
        self.plot_item_to_render_part_key: dict[pg.PlotDataItem, str] = {}
        self._update_scheduled = False
        self._interval_indices = self._normalize_interval_indices(
            interval_indices if interval_indices is not None else (0, self.point_count - 1)
        )
        self._excluded_indices: set[int] = set(excluded_indices)

        self.legend = self.plot_widget.addLegend()

        self.inactive_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(130, 130, 130, 110),
            size=6,
        )
        self.active_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(220, 60, 60, 180),
            size=7,
        )
        self.inactive_scatter.sigClicked.connect(self._handle_scatter_clicked)
        self.active_scatter.sigClicked.connect(self._handle_scatter_clicked)
        self.plot_widget.addItem(self.inactive_scatter)
        self.plot_widget.addItem(self.active_scatter)
        self.selection_label = QLabel("")
        layout.addWidget(self.selection_label)

        # Line plot for model
        self.x_line = self._get_data_x_line()
        self.line_plot = self.plot_widget.plot(
            self.x_line,
            evaluate_model(self.model, self.x_line, *order_initial_params(self.model, self.params)),
            pen=pg.mkPen('g', width=2),
            name="Model",
        )

        for idx, part in enumerate(self.render_parts):
            pen = pg.mkPen(part.color or DEFAULT_RENDER_PART_COLORS[idx % len(DEFAULT_RENDER_PART_COLORS)], width=2, style=Qt.PenStyle.DashLine)
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
        self._refresh_scatter_points()
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

    def _normalize_interval_indices(
        self,
        interval_indices: tuple[int, int],
    ) -> tuple[int, int]:
        start_idx, end_idx = sorted(interval_indices)
        start_idx = max(0, min(start_idx, self.point_count - 1))
        end_idx = max(0, min(end_idx, self.point_count - 1))
        return (start_idx, end_idx)

    def get_interval_indices(self) -> tuple[int, int] | None:
        return self._interval_indices

    def get_excluded_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._excluded_indices))

    def close_all_windows(self):
        self.close()
        if self.param_win is not None:
            self.param_win.close()
        if self.fit_selection_win is not None:
            self.fit_selection_win.close()

    def get_active_mask(self) -> np.ndarray:
        mask = np.ones(self.point_count, dtype=bool)
        start_idx, end_idx = self._interval_indices
        interval_mask = np.zeros(self.point_count, dtype=bool)
        interval_mask[start_idx:end_idx + 1] = True
        mask &= interval_mask
        if self._excluded_indices:
            excluded = np.array(sorted(self._excluded_indices), dtype=int)
            valid = excluded[(excluded >= 0) & (excluded < self.point_count)]
            mask[valid] = False
        return mask

    def _build_scatter_spots(self, indices: np.ndarray):
        return [
            {
                "pos": (float(self.x_data[idx]), float(self.y_data[idx])),
                "data": int(idx),
            }
            for idx in indices
        ]

    def _refresh_scatter_points(self):
        active_mask = self.get_active_mask()
        active_indices = np.flatnonzero(active_mask)
        inactive_indices = np.flatnonzero(~active_mask)
        self.active_scatter.setData(spots=self._build_scatter_spots(active_indices))
        self.inactive_scatter.setData(spots=self._build_scatter_spots(inactive_indices))
        self._update_selection_label()

    def _update_selection_label(self):
        active_count = int(np.count_nonzero(self.get_active_mask()))
        total_count = self.point_count
        interval_text = f"[{self._interval_indices[0]}, {self._interval_indices[1]}]"
        self.selection_label.setText(
            f"Active points: {active_count}/{total_count} | Interval: {interval_text} | Excluded: {len(self._excluded_indices)}"
        )
        if self.fit_selection_win is not None:
            self.fit_selection_win.sync_fit_selection_controls(
                interval_indices=self._interval_indices,
            )
            self.fit_selection_win.update_selection_summary(
                active_count=active_count,
                total_count=total_count,
                interval_indices=self.get_interval_indices(),
                excluded_count=len(self._excluded_indices),
                excluded_indices=self.get_excluded_indices(),
            )

    def set_interval_indices(self, start_idx: int, end_idx: int):
        self._interval_indices = self._normalize_interval_indices((start_idx, end_idx))
        self._refresh_scatter_points()

    def set_excluded_indices(self, excluded_indices: tuple[int, ...] | list[int] | set[int]):
        self._excluded_indices = {
            idx for idx in excluded_indices
            if 0 <= idx < self.point_count
        }
        self._refresh_scatter_points()

    def clear_excluded_indices(self):
        self._excluded_indices.clear()
        self._refresh_scatter_points()

    def _handle_scatter_clicked(self, _plot, points):
        changed = False
        for point in points:
            idx = point.data()
            if idx is None:
                continue
            idx = int(idx)
            if idx in self._excluded_indices:
                self._excluded_indices.remove(idx)
            else:
                self._excluded_indices.add(idx)
            changed = True
        if changed:
            self._refresh_scatter_points()

    def update_visible_model_curve(self, *_args):
        self.x_line = self._get_visible_x_line()
        y_line = evaluate_model(self.model, self.x_line, *order_initial_params(self.model, self.params))
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
            self.close_all_windows()
        else:
            super().keyPressEvent(a0)

    def closeEvent(self, a0) -> None:
        callback = self.on_close_callback
        super().closeEvent(a0)
        if callable(callback):
            callback()
