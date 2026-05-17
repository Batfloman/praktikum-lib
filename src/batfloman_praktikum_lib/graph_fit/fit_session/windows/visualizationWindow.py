from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...init_params._helper import smart_format
from ..session import FitSession


class FitSessionVisualizationWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, session: FitSession, *, window_title: str = "Fit Session Visualization"):
        super().__init__()
        self.session = session
        self.models_window = None
        self._is_closing_all = False
        self._has_initialized_view = False
        self._interval_items: list[pg.LinearRegionItem] = []
        self._curve_items: list[pg.PlotDataItem] = []
        self._band_items: list[pg.GraphicsObject] = []
        self._model_plot_items: dict[object, int] = {}
        self._close_shortcut = QShortcut(QKeySequence("Q"), self)
        self._close_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._close_shortcut.activated.connect(self.close_all_windows)

        self.setWindowTitle(window_title)
        self.resize(1100, 800)

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.legend = self.plot_widget.addLegend()
        layout.addWidget(self.plot_widget)

        self.coord_label = QLabel("x=0, y=0")
        layout.addWidget(self.coord_label)

        self.x_data = np.asarray(self.session.x, dtype=float)
        self.y_data = np.asarray(self.session.y, dtype=float)
        self.plot_widget.scene().sigMouseMoved.connect(self._mouse_moved)

        self.refresh(preserve_view=False)

    def refresh(self, *, preserve_view: bool = True):
        previous_view = None
        if preserve_view and self._has_initialized_view:
            previous_view = self.plot_widget.plotItem.vb.viewRange()

        self.plot_widget.clear()
        self.legend = self.plot_widget.addLegend()
        self.data_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(45, 55, 72, 180),
            size=6,
            name="Data",
        )
        self.plot_widget.addItem(self.data_scatter)
        self._interval_items.clear()
        self._curve_items.clear()
        self._band_items.clear()
        self._model_plot_items.clear()
        self._refresh_data()
        self._refresh_models()

        if previous_view is None:
            self.plot_widget.enableAutoRange()
            self._has_initialized_view = True
            return

        self.plot_widget.plotItem.disableAutoRange()
        self.plot_widget.setXRange(previous_view[0][0], previous_view[0][1], padding=0)
        self.plot_widget.setYRange(previous_view[1][0], previous_view[1][1], padding=0)

    def close_all_windows(self):
        if self._is_closing_all:
            return
        self._is_closing_all = True
        self.close()
        if self.models_window is not None and self.models_window.isVisible():
            self.models_window.close()

    def _refresh_data(self):
        spots = [
            {
                "pos": (float(x_val), float(y_val)),
                "data": idx,
            }
            for idx, (x_val, y_val) in enumerate(zip(self.x_data, self.y_data))
        ]
        self.data_scatter.setData(spots=spots)

    def _refresh_models(self):
        selected_model_id = None
        if self.models_window is not None:
            selected_model_id = self.models_window._selected_model_id()

        full_xmin, full_xmax = self._full_x_range()
        for instance in self.session.models:
            if not instance.visible:
                continue

            color = pg.intColor(instance.color_index, hues=max(1, len(self.session.models)))
            interval = self.session._resolve_x_interval(instance)
            is_selected = instance.id == selected_model_id

            if self._should_show_interval(instance, is_selected=is_selected):
                region = pg.LinearRegionItem(
                    values=interval,
                    orientation="vertical",
                    movable=False,
                    brush=pg.mkBrush(color.red(), color.green(), color.blue(), 30),
                    pen=pg.mkPen(color=color, width=1),
                )
                self.plot_widget.addItem(region)
                region.setZValue(10)
                self._register_clickable_item(region, instance.id)
                self._interval_items.append(region)

            if instance.result is None:
                continue

            if is_selected:
                self._plot_selected_model(instance, color, interval, full_xmin, full_xmax)
                continue

            x_line = np.linspace(interval[0], interval[1], 1000)
            y_line = np.asarray(instance.result.func_no_err(x_line), dtype=float)
            curve = self.plot_widget.plot(
                x_line,
                y_line,
                pen=pg.mkPen(color=color, width=2),
                name=instance.display_name,
            )
            curve.setCurveClickable(True, width=12)
            self._register_clickable_item(curve, instance.id)
            self._curve_items.append(curve)

    def _plot_selected_model(
        self,
        instance,
        color,
        interval: tuple[float, float],
        full_xmin: float,
        full_xmax: float,
    ) -> None:
        interval_x = np.linspace(interval[0], interval[1], 1000)
        interval_y = np.asarray(instance.result.func_no_err(interval_x), dtype=float)
        main_curve = self.plot_widget.plot(
            interval_x,
            interval_y,
            pen=pg.mkPen(color=color, width=3),
            name=instance.display_name,
        )
        main_curve.setCurveClickable(True, width=12)
        self._register_clickable_item(main_curve, instance.id)
        self._curve_items.append(main_curve)

        if full_xmin < interval[0]:
            self._plot_muted_segment(instance, full_xmin, interval[0], color)
        if interval[1] < full_xmax:
            self._plot_muted_segment(instance, interval[1], full_xmax, color)

        if instance.show_1sigma_band:
            lower = np.asarray(instance.result.min_1sigma(interval_x), dtype=float)
            upper = np.asarray(instance.result.max_1sigma(interval_x), dtype=float)
            lower_curve = pg.PlotCurveItem(
                interval_x,
                lower,
                pen=pg.mkPen(color=color, width=1),
            )
            upper_curve = pg.PlotCurveItem(
                interval_x,
                upper,
                pen=pg.mkPen(color=color, width=1),
            )
            band = pg.FillBetweenItem(
                lower_curve,
                upper_curve,
                brush=pg.mkBrush(color.red(), color.green(), color.blue(), 45),
            )
            self.plot_widget.addItem(lower_curve)
            self.plot_widget.addItem(upper_curve)
            self.plot_widget.addItem(band)
            self._curve_items.extend([lower_curve, upper_curve])
            self._band_items.append(band)

    def _plot_muted_segment(self, instance, xmin: float, xmax: float, color) -> None:
        if xmax <= xmin:
            return
        x_line = np.linspace(xmin, xmax, 400)
        y_line = np.asarray(instance.result.func_no_err(x_line), dtype=float)
        curve = self.plot_widget.plot(
            x_line,
            y_line,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
        )
        curve.setCurveClickable(True, width=12)
        self._register_clickable_item(curve, instance.id)
        curve.setOpacity(0.35)
        self._curve_items.append(curve)

    def _register_clickable_item(self, item, model_id: int) -> None:
        self._model_plot_items[item] = model_id
        if hasattr(item, "sigClicked"):
            item.sigClicked.connect(
                lambda clicked_item, _points=None, item=item: self._handle_plot_item_clicked(item)
            )

    def _handle_plot_item_clicked(self, item) -> None:
        if self.models_window is None:
            return
        model_id = self._model_plot_items.get(item)
        if model_id is None:
            return
        self.models_window.select_model(model_id)

    def _should_show_interval(self, instance, *, is_selected: bool) -> bool:
        if instance.interval_display_mode == "off":
            return False
        if instance.interval_display_mode == "always":
            return True
        return is_selected

    def _full_x_range(self) -> tuple[float, float]:
        if self.x_data.size == 0:
            return (0.0, 0.0)
        return (float(np.min(self.x_data)), float(np.max(self.x_data)))

    def _mouse_moved(self, pos):
        vb = self.plot_widget.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        self.coord_label.setText(
            f"x={smart_format(mouse_point.x())}, y={smart_format(mouse_point.y())}"
        )

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        if a0 is not None and a0.key() == Qt.Key.Key_Q:
            self.close_all_windows()
            return
        super().keyPressEvent(a0)

    def closeEvent(self, a0) -> None:
        self.closed.emit()
        if self.models_window is not None and self.models_window.isVisible() and not self._is_closing_all:
            self.models_window.close()
        super().closeEvent(a0)
