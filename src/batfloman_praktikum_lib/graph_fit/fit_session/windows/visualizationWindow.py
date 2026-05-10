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

        self.refresh()

    def refresh(self):
        previous_view = None
        if self._has_initialized_view:
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
        for instance in self.session.models:
            if not instance.visible:
                continue

            color = pg.intColor(instance.color_index, hues=max(1, len(self.session.models)))
            interval = self.session._resolve_x_interval(instance)
            region = pg.LinearRegionItem(
                values=interval,
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(color.red(), color.green(), color.blue(), 30),
                pen=pg.mkPen(None),
            )
            self.plot_widget.addItem(region)
            region.setZValue(-10)
            self._interval_items.append(region)

            if instance.result is None:
                continue

            x_line = np.linspace(interval[0], interval[1], 1000)
            y_line = np.asarray(instance.result.func_no_err(x_line), dtype=float)
            curve = self.plot_widget.plot(
                x_line,
                y_line,
                pen=pg.mkPen(color=color, width=2),
                name=instance.display_name,
            )
            self._curve_items.append(curve)

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
