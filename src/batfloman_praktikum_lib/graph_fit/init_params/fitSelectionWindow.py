import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ._helper import smart_format


class CollapsibleGroup(QWidget):
    def __init__(self, title: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toggle = QToolButton(text=title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.toggled.connect(self._set_expanded)
        layout.addWidget(self.toggle)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 0, 0, 0)
        layout.addWidget(self.content)

    def _set_expanded(self, expanded: bool):
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class ClampedIndexSpinBox(QSpinBox):
    def validate(self, text: str, pos: int):
        stripped = text.strip()
        if stripped == "":
            return (QValidator.State.Intermediate, text, pos)
        if stripped.isdigit():
            return (QValidator.State.Acceptable, text, pos)
        return (QValidator.State.Invalid, text, pos)

    def fixup(self, text: str) -> str:
        stripped = text.strip()
        if stripped == "":
            return str(self.minimum())
        try:
            value = int(stripped)
        except ValueError:
            return str(self.minimum())
        value = max(self.minimum(), min(value, self.maximum()))
        return str(value)

    def valueFromText(self, text: str) -> int:
        stripped = text.strip()
        if stripped == "":
            return self.minimum()
        try:
            value = int(stripped)
        except ValueError:
            return self.minimum()
        return max(self.minimum(), min(value, self.maximum()))


class FitSelectionWindow(QWidget):
    graph_win = None

    def __init__(
        self,
        *,
        point_count: int,
        interval_indices: tuple[int, int] | None = None,
        excluded_indices: tuple[int, ...] = (),
        filtered_out_count: int = 0,
        x_preview=None,
        y_preview=None,
    ):
        super().__init__()

        self.setWindowTitle("Fit Selection")
        self.point_count = point_count
        self.filtered_out_count = filtered_out_count
        self.x_preview = np.asarray(x_preview, dtype=float) if x_preview is not None else None
        self.y_preview = np.asarray(y_preview, dtype=float) if y_preview is not None else None
        self._syncing = False

        self.interval_start_spin: QSpinBox | None = None
        self.interval_end_spin: QSpinBox | None = None
        self.selection_summary_label: QLabel | None = None
        self.exclusion_summary_label: QLabel | None = None
        self.excluded_points_label: QLabel | None = None

        outer_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        scroll.setWidget(content)

        self._build_selection_layout(
            interval_indices=interval_indices,
            excluded_indices=excluded_indices,
        )
        self.layout.addStretch(1)

    def _build_selection_layout(
        self,
        *,
        interval_indices: tuple[int, int] | None,
        excluded_indices: tuple[int, ...],
    ):
        group_widget = CollapsibleGroup("Fit Selection")
        panel = QFrame()
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setHorizontalSpacing(8)
        panel_layout.setVerticalSpacing(6)

        start_idx, end_idx = interval_indices or (0, self.point_count - 1)

        filter_notice_label = QLabel(self._get_filter_notice_text())
        filter_notice_label.setWordWrap(True)
        panel_layout.addWidget(filter_notice_label, 0, 0, 1, 3)

        panel_layout.addWidget(QLabel("Start"), 1, 0)
        self.interval_start_spin = ClampedIndexSpinBox()
        self.interval_start_spin.setRange(0, self.point_count - 1)
        self.interval_start_spin.setKeyboardTracking(False)
        self.interval_start_spin.setValue(start_idx)
        self.interval_start_spin.valueChanged.connect(self._handle_interval_spin_changed)
        panel_layout.addWidget(self.interval_start_spin, 1, 1)

        panel_layout.addWidget(QLabel("End"), 2, 0)
        self.interval_end_spin = ClampedIndexSpinBox()
        self.interval_end_spin.setRange(0, self.point_count - 1)
        self.interval_end_spin.setKeyboardTracking(False)
        self.interval_end_spin.setValue(end_idx)
        self.interval_end_spin.valueChanged.connect(self._handle_interval_spin_changed)
        panel_layout.addWidget(self.interval_end_spin, 2, 1)

        full_range_button = QPushButton("Use all")
        full_range_button.clicked.connect(self._use_full_index_range)
        panel_layout.addWidget(full_range_button, 1, 2, 2, 1)

        self.selection_summary_label = QLabel("")
        panel_layout.addWidget(self.selection_summary_label, 3, 0, 1, 3)

        self.exclusion_summary_label = QLabel("")
        panel_layout.addWidget(self.exclusion_summary_label, 4, 0, 1, 3)

        helper_label = QLabel("Click points in the graph to toggle exclusion.")
        panel_layout.addWidget(helper_label, 5, 0, 1, 3)

        excluded_group = CollapsibleGroup("Excluded Points")
        reenable_button = QPushButton("Re-enable all")
        reenable_button.clicked.connect(self._reenable_all_points)
        excluded_group.content_layout.addWidget(reenable_button)
        self.excluded_points_label = QLabel("")
        self.excluded_points_label.setWordWrap(True)
        excluded_group.content_layout.addWidget(self.excluded_points_label)

        group_widget.content_layout.addWidget(panel)
        group_widget.content_layout.addWidget(excluded_group)
        self.layout.addWidget(group_widget)

        self.update_selection_summary(
            active_count=self.point_count - len(excluded_indices),
            total_count=self.point_count,
            interval_indices=interval_indices,
            excluded_count=len(excluded_indices),
            excluded_indices=excluded_indices,
        )

    def _get_filter_notice_text(self) -> str:
        if self.filtered_out_count <= 0:
            return "Indices refer to the plotted points in their current order."
        return (
            f"Indices refer to the plotted points after NaN filtering. "
            f"Removed before selection: {self.filtered_out_count} point(s)."
        )

    def _format_excluded_points_text(self, excluded_indices: tuple[int, ...]) -> str:
        if not excluded_indices:
            return "No excluded points."
        if self.x_preview is None or self.y_preview is None:
            return "Excluded indices: " + ", ".join(str(idx) for idx in excluded_indices)

        lines = []
        for idx in excluded_indices:
            if idx < 0 or idx >= len(self.x_preview):
                continue
            lines.append(
                f"{idx}: x={smart_format(self.x_preview[idx])}, y={smart_format(self.y_preview[idx])}"
            )
        return "\n".join(lines) if lines else "No excluded points."

    def _use_full_index_range(self):
        if self.interval_start_spin is None or self.interval_end_spin is None:
            return
        self.interval_start_spin.setValue(0)
        self.interval_end_spin.setValue(self.point_count - 1)

    def _handle_interval_controls_changed(self, *_args):
        if self._syncing or self.graph_win is None:
            return
        start_idx = self.interval_start_spin.value() if self.interval_start_spin is not None else 0
        end_idx = self.interval_end_spin.value() if self.interval_end_spin is not None else self.point_count - 1
        self.graph_win.set_interval_indices(start_idx, end_idx)

    def _handle_interval_spin_changed(self, *_args):
        if self._syncing:
            return
        self._handle_interval_controls_changed()

    def _reenable_all_points(self):
        if self.graph_win is None:
            return
        self.graph_win.clear_excluded_indices()

    def update_selection_summary(
        self,
        *,
        active_count: int,
        total_count: int,
        interval_indices: tuple[int, int] | None,
        excluded_count: int,
        excluded_indices: tuple[int, ...] = (),
    ):
        if self.selection_summary_label is not None:
            interval_text = "all indices" if interval_indices is None else f"[{interval_indices[0]}, {interval_indices[1]}]"
            self.selection_summary_label.setText(
                f"Using {active_count}/{total_count} points | Interval: {interval_text}"
            )
        if self.exclusion_summary_label is not None:
            self.exclusion_summary_label.setText(f"Excluded points: {excluded_count}")
        if self.excluded_points_label is not None:
            self.excluded_points_label.setText(
                self._format_excluded_points_text(excluded_indices)
            )

    def sync_fit_selection_controls(
        self,
        *,
        interval_indices: tuple[int, int],
    ):
        self._syncing = True
        if self.interval_start_spin is not None:
            self.interval_start_spin.setValue(interval_indices[0])
        if self.interval_end_spin is not None:
            self.interval_end_spin.setValue(interval_indices[1])
        self._syncing = False

    def keyPressEvent(self, a0) -> None:
        if a0.key() == Qt.Key.Key_Q:
            if self.graph_win is not None:
                self.graph_win.close_all_windows()
            else:
                self.close()
        else:
            super().keyPressEvent(a0)
