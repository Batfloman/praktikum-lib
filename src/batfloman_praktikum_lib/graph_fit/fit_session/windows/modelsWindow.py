from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from ...fitResult import format_fit_quality
from ..session import (
    AvailableModels,
    CompositionComponent,
    FitSession,
    IntervalDisplayMode,
    SessionModel,
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


class ClampedFloatSpinBox(QDoubleSpinBox):
    def validate(self, text: str, pos: int):
        stripped = text.strip()
        if stripped in {"", "-", "+", ".", "-.", "+."}:
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(stripped)
        except ValueError:
            return (QValidator.State.Invalid, text, pos)
        return (QValidator.State.Acceptable, text, pos)

    def fixup(self, text: str) -> str:
        stripped = text.strip()
        if stripped == "":
            return f"{self.minimum():g}"
        try:
            value = float(stripped)
        except ValueError:
            return f"{self.minimum():g}"
        value = max(self.minimum(), min(value, self.maximum()))
        return f"{value:g}"

    def valueFromText(self, text: str) -> float:
        stripped = text.strip()
        if stripped == "":
            return self.minimum()
        try:
            value = float(stripped)
        except ValueError:
            return self.minimum()
        return max(self.minimum(), min(value, self.maximum()))


class ModelTreeWidget(QTreeWidget):
    pass


class InlineRenameEdit(QLineEdit):
    focus_lost = pyqtSignal()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.focus_lost.emit()


class AutoPopupComboBox(QComboBox):
    def mousePressEvent(self, event) -> None:
        should_open = (
            event.button() == Qt.MouseButton.LeftButton
            and not self.view().isVisible()
        )
        super().mousePressEvent(event)
        if should_open:
            QTimer.singleShot(0, self.showPopup)

    def wheelEvent(self, event) -> None:
        event.ignore()


class FitSessionModelsWindow(QWidget):
    closed = pyqtSignal()

    def __init__(
        self,
        session: FitSession,
        *,
        default_model=None,
        available_models: AvailableModels | None = None,
        window_title: str = "Fit Session Models",
    ):
        super().__init__()
        self.session = session
        self.visualization_window = None
        self.default_model = default_model
        self.available_models = self._resolve_available_models(
            default_model=default_model,
            available_models=available_models,
        )
        self._updating_tree = False
        self._updating_interval_controls = False
        self._is_closing_all = False
        self._active_rename: tuple[QTreeWidgetItem, str, tuple[Any, ...]] | None = None
        self._pending_autofit_model_id: int | None = None
        self._autofit_timer = QTimer(self)
        self._autofit_timer.setSingleShot(True)
        self._autofit_timer.setInterval(250)
        self._autofit_timer.timeout.connect(self._run_pending_autofit)
        self._close_shortcut = QShortcut(QKeySequence("Q"), self)
        self._close_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._close_shortcut.activated.connect(self.close_all_windows)

        self.setWindowTitle(window_title)
        self.resize(620, 920)

        layout = QVBoxLayout(self)

        add_model_row = QHBoxLayout()
        self.add_model_button = QPushButton("Add Model")
        self.add_model_button.clicked.connect(self._handle_add_model)
        add_model_row.addWidget(self.add_model_button)
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(self._handle_move_up)
        add_model_row.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(self._handle_move_down)
        add_model_row.addWidget(self.move_down_button)
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self._handle_remove_selected)
        add_model_row.addWidget(self.remove_selected_button)
        layout.addLayout(add_model_row)

        self.model_tree = ModelTreeWidget()
        self.model_tree.setColumnCount(4)
        self.model_tree.setHeaderLabels(["Model", "State", "Details", "Status"])
        self.model_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_tree.itemSelectionChanged.connect(self._handle_selection_changed)
        self.model_tree.itemChanged.connect(self._handle_item_changed)
        self.model_tree.itemDoubleClicked.connect(self._handle_item_double_clicked)
        layout.addWidget(self.model_tree, stretch=1)

        interval_box = QWidget()
        self.interval_layout = QFormLayout(interval_box)
        self.interval_kind_selector = QComboBox()
        self.interval_kind_selector.addItem("Indices", userData="index")
        self.interval_kind_selector.addItem("X Values", userData="x")
        self.interval_kind_selector.currentIndexChanged.connect(self._handle_interval_kind_changed)
        self.interval_layout.addRow("Interval Type", self.interval_kind_selector)

        self.interval_start_spin = ClampedIndexSpinBox()
        self.interval_end_spin = ClampedIndexSpinBox()
        max_index = max(0, len(self.session.x) - 1)
        self.interval_start_spin.setRange(0, max_index)
        self.interval_end_spin.setRange(0, max_index)
        self.interval_start_spin.setKeyboardTracking(False)
        self.interval_end_spin.setKeyboardTracking(False)
        self.interval_start_spin.valueChanged.connect(self._handle_interval_value_changed)
        self.interval_end_spin.valueChanged.connect(self._handle_interval_value_changed)
        self.interval_layout.addRow("Start Index", self.interval_start_spin)
        self.interval_layout.addRow("End Index", self.interval_end_spin)

        x_values = np.asarray(self.session.x, dtype=float)
        x_min = float(np.min(x_values)) if len(x_values) > 0 else 0.0
        x_max = float(np.max(x_values)) if len(x_values) > 0 else 0.0
        x_step = abs(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
        self.x_interval_start_spin = ClampedFloatSpinBox()
        self.x_interval_end_spin = ClampedFloatSpinBox()
        for spin in (self.x_interval_start_spin, self.x_interval_end_spin):
            spin.setDecimals(6)
            spin.setRange(x_min, x_max)
            spin.setSingleStep(x_step if x_step > 0 else 1.0)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._handle_interval_value_changed)
        self.interval_layout.addRow("Start X", self.x_interval_start_spin)
        self.interval_layout.addRow("End X", self.x_interval_end_spin)

        interval_buttons = QHBoxLayout()
        self.reset_interval_button = QPushButton("Full Range")
        self.reset_interval_button.clicked.connect(self._handle_reset_interval)
        interval_buttons.addWidget(self.reset_interval_button)
        self.interval_layout.addRow(interval_buttons)
        self.auto_refit_checkbox = QCheckBox("Auto-refit on interval change")
        self.interval_layout.addRow(self.auto_refit_checkbox)
        layout.addWidget(interval_box)

        component_box = QWidget()
        component_layout = QHBoxLayout(component_box)
        self.component_selector = QComboBox()
        for name in self.available_models:
            self.component_selector.addItem(name)
        component_layout.addWidget(self.component_selector, stretch=1)
        self.add_component_button = QPushButton("Add Composition Model")
        self.add_component_button.clicked.connect(self._handle_add_component)
        component_layout.addWidget(self.add_component_button)
        layout.addWidget(component_box)

        controls_row = QHBoxLayout()
        self.edit_parameters_button = QPushButton("Edit Parameters")
        self.edit_parameters_button.clicked.connect(self._handle_edit_parameters)
        controls_row.addWidget(self.edit_parameters_button)
        self.refit_selected_button = QPushButton("Refit Selected Model")
        self.refit_selected_button.clicked.connect(self._handle_refit_selected)
        controls_row.addWidget(self.refit_selected_button)
        layout.addLayout(controls_row)

        fit_row = QHBoxLayout()
        self.fit_all_button = QPushButton("Fit All")
        self.fit_all_button.clicked.connect(self._handle_fit_all)
        fit_row.addWidget(self.fit_all_button)
        layout.addLayout(fit_row)

        self.status_label = QLabel("No models loaded.")
        layout.addWidget(self.status_label)

        self._sync_interval_input_visibility("index")
        self.refresh()

    def refresh(self, *, select_model_id: Optional[int] = None):
        self._refresh_model_list(select_model_id=select_model_id)
        if self.visualization_window is not None:
            self.visualization_window.refresh()

    def _expanded_component_ids(self) -> set[tuple[int, int]]:
        expanded: set[tuple[int, int]] = set()
        for row in range(self.model_tree.topLevelItemCount()):
            model_item = self.model_tree.topLevelItem(row)
            for child_idx in range(model_item.childCount()):
                child = model_item.child(child_idx)
                child_type = child.data(0, Qt.ItemDataRole.UserRole)
                if child_type is None or child_type[0] != "composition":
                    continue
                for component_idx in range(child.childCount()):
                    component_item = child.child(component_idx)
                    component_type = component_item.data(0, Qt.ItemDataRole.UserRole)
                    if (
                        component_type is not None
                        and component_type[0] == "component"
                        and component_item.isExpanded()
                    ):
                        expanded.add((int(component_type[1]), int(component_type[2])))
        return expanded

    def _create_interval_mode_combo(
        self,
        model_id: int,
        current_mode: IntervalDisplayMode,
    ) -> QComboBox:
        combo = AutoPopupComboBox(self.model_tree)
        combo.addItem("off", userData="off")
        combo.addItem("selected-only", userData="selected-only")
        combo.addItem("always", userData="always")
        combo.setCurrentIndex(combo.findData(current_mode))
        combo.currentIndexChanged.connect(
            lambda _index, model_id=model_id, combo=combo: self._handle_interval_mode_changed(
                model_id,
                combo,
            )
        )
        return combo

    def close_all_windows(self):
        if self._is_closing_all:
            return
        self._is_closing_all = True
        self.close()
        if self.visualization_window is not None and self.visualization_window.isVisible():
            self.visualization_window.close()

    def _default_available_models(self, default_model):
        from ...models import ConstFunc, Exponential, Gaussian, Linear, LinearShifted

        models = {
            "Gaussian": Gaussian,
            "Linear": Linear,
            "LinearShifted": LinearShifted,
            "Exponential": Exponential,
            "ConstFunc": ConstFunc,
        }
        if default_model is not None:
            models = {default_model.__name__: default_model, **models}
        return models

    def _resolve_available_models(
        self,
        *,
        default_model,
        available_models: AvailableModels | None,
    ) -> dict[str, Any]:
        models = self._default_available_models(default_model)
        if available_models is None:
            return models
        return {**models, **available_models}

    def _handle_add_model(self):
        model_id = self.session.add_model(
            interval=(0, len(self.session.x) - 1),
            interval_kind="index",
            name=f"Model {len(self.session.models) + 1}",
        )
        if self.default_model is not None:
            self.session.add_component(model_id, self.default_model)
        self.refresh(select_model_id=model_id)

    def _handle_remove_selected(self):
        selected_model = self._selected_model()
        if selected_model is None:
            return

        selected_component = self._selected_component()
        if selected_component is not None:
            self.session.remove_component(selected_model.id, selected_component.id)
            self.refresh(select_model_id=selected_model.id)
            return

        self.session.remove_model(selected_model.id)
        self.refresh()

    def _handle_add_component(self):
        selected_model = self._selected_model()
        if selected_model is None:
            return
        component_name = self.component_selector.currentText()
        component_model = self.available_models[component_name]
        self.session.add_component(selected_model.id, component_model, name=component_name)
        self.refresh(select_model_id=selected_model.id)

    def _handle_move_up(self):
        self._move_selected(-1)

    def _handle_move_down(self):
        self._move_selected(1)

    def _handle_fit_all(self):
        if not self.session.models:
            return
        try:
            self.session.fit()
        except Exception as exc:
            self._show_error("Fit All failed", exc)
            self.refresh(select_model_id=self._selected_model_id())
            return
        self.refresh(select_model_id=self._selected_model_id())

    def _handle_refit_selected(self):
        selected_model = self._selected_model()
        if selected_model is None:
            return
        try:
            self.session.fit_model(selected_model.id)
        except Exception as exc:
            self._show_error(f"Refit failed for {selected_model.display_name}", exc)
            self.refresh(select_model_id=selected_model.id)
            return
        self.refresh(select_model_id=selected_model.id)

    def _handle_edit_parameters(self):
        selected_model = self._selected_model()
        if selected_model is None:
            return
        try:
            self.session.open_parameter_editor(selected_model.id)
        except Exception as exc:
            self._show_error(f"Parameter editor failed for {selected_model.display_name}", exc)
            self.refresh(select_model_id=selected_model.id)
            return
        self.refresh(select_model_id=selected_model.id)

    def _handle_interval_value_changed(self):
        if self._updating_interval_controls:
            return
        selected_model = self._selected_model()
        if selected_model is None:
            return
        interval_kind = self._selected_interval_kind()
        if interval_kind == "index":
            start_value = self.interval_start_spin.value()
            end_value = self.interval_end_spin.value()
        else:
            start_value = self.x_interval_start_spin.value()
            end_value = self.x_interval_end_spin.value()
        new_interval = tuple(sorted((start_value, end_value)))
        if selected_model.interval_kind == interval_kind and selected_model.interval == new_interval:
            return
        self.session.set_interval(selected_model.id, new_interval, interval_kind=interval_kind)
        self.refresh(select_model_id=selected_model.id)
        if self.auto_refit_checkbox.isChecked():
            self._schedule_autofit(selected_model.id)

    def _handle_interval_kind_changed(self):
        if self._updating_interval_controls:
            return
        selected_model = self._selected_model()
        interval_kind = self._selected_interval_kind()
        self._sync_interval_input_visibility(interval_kind)
        if selected_model is None:
            return

        previous_kind = self.session._resolve_interval_kind(selected_model)
        if interval_kind == previous_kind:
            return

        if interval_kind == "index":
            interval = self._convert_interval_to_index(selected_model)
        else:
            interval = self._convert_interval_to_x(selected_model)

        self.session.set_interval(selected_model.id, interval, interval_kind=interval_kind)
        self.refresh(select_model_id=selected_model.id)
        if self.auto_refit_checkbox.isChecked():
            self._schedule_autofit(selected_model.id)

    def _handle_reset_interval(self):
        selected_model = self._selected_model()
        if selected_model is None:
            return
        interval_kind = self._selected_interval_kind()
        if interval_kind == "index":
            interval = (0, len(self.session.x) - 1)
        else:
            interval = self._full_x_interval()
        self.session.set_interval(selected_model.id, interval, interval_kind=interval_kind)
        self.refresh(select_model_id=selected_model.id)
        if self.auto_refit_checkbox.isChecked():
            self._schedule_autofit(selected_model.id)

    def _handle_selection_changed(self):
        selected_model = self._selected_model()
        selected_component = self._selected_component()
        has_model = selected_model is not None
        has_component = selected_component is not None

        self.remove_selected_button.setEnabled(has_model)
        self.move_up_button.setEnabled(has_model)
        self.move_down_button.setEnabled(has_model)
        self.add_component_button.setEnabled(has_model)
        self.edit_parameters_button.setEnabled(has_model)
        self.refit_selected_button.setEnabled(has_model)
        self.reset_interval_button.setEnabled(has_model)

        if selected_model is None:
            self.status_label.setText("No model selected.")
            return

        self._updating_interval_controls = True
        resolved_kind = self.session._resolve_interval_kind(selected_model)
        self.interval_kind_selector.setCurrentIndex(
            self.interval_kind_selector.findData(resolved_kind)
        )
        self._sync_interval_input_visibility(resolved_kind)

        index_interval = self._effective_index_interval(selected_model)
        self.interval_start_spin.setValue(index_interval[0])
        self.interval_end_spin.setValue(index_interval[1])

        x_interval = self.session._resolve_x_interval(selected_model)
        self.x_interval_start_spin.setValue(x_interval[0])
        self.x_interval_end_spin.setValue(x_interval[1])
        self._updating_interval_controls = False

        if has_component:
            component_status = selected_component.load_error or "ok"
            self.status_label.setText(
                f"Selected component: {selected_component.display_name} in {selected_model.display_name} | {component_status}"
            )
            self._update_interval_render_statuses(selected_model_id=selected_model.id)
            if self.visualization_window is not None:
                self.visualization_window.refresh()
            return

        self.status_label.setText(
            f"Selected: {selected_model.display_name} | Status: {self._status_text(selected_model)}"
        )
        if selected_model.load_warning is not None:
            self.status_label.setText(
                f"Selected: {selected_model.display_name} | Warning: {selected_model.load_warning}"
            )
        if selected_model.last_error is not None:
            self.status_label.setText(
                f"Selected: {selected_model.display_name} | Failed: {selected_model.last_error}"
            )
        self._update_interval_render_statuses(selected_model_id=selected_model.id)
        if self.visualization_window is not None:
            self.visualization_window.refresh()

    def _handle_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._updating_tree:
            return

        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        if item_type is None:
            return

        if item_type[0] == "model":
            if column != 0:
                return
            model_id = item_type[1]
            visible = item.checkState(0) == Qt.CheckState.Checked
            instance = self.session.get_model(model_id)
            if instance.visible != visible:
                self.session.set_visible(model_id, visible)
                if self.visualization_window is not None:
                    self.visualization_window.refresh()
            return

        if item_type[0] == "fit_quality":
            if column != 2:
                return
            model_id = item_type[1]
            enabled = item.checkState(2) == Qt.CheckState.Checked
            instance = self.session.get_model(model_id)
            if instance.show_1sigma_band != enabled:
                self.session.set_show_1sigma_band(model_id, enabled)
                QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))
            return

        if item_type[0] == "component":
            if column != 0:
                return
            model_id = item_type[1]
            component_id = item_type[2]
            enabled = item.checkState(0) == Qt.CheckState.Checked
            component = self.session.get_component(model_id, component_id)
            if component.enabled != enabled:
                try:
                    self.session.set_component_enabled(model_id, component_id, enabled)
                except ValueError as exc:
                    self._show_error("Component toggle failed", exc)
                    QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))
                    return
                QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))
                return

    def _handle_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._active_rename is not None:
            return
        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        if item_type is None or item_type[0] not in {"model", "component"}:
            return
        self._start_inline_rename(item, item_type)

    def _start_inline_rename(self, item: QTreeWidgetItem, item_type: tuple[Any, ...]) -> None:
        original_text = item.text(0)
        editor = InlineRenameEdit(original_text, self.model_tree)
        editor.setFrame(False)
        editor.setStyleSheet(
            "QLineEdit {"
            " border: 1px solid palette(mid);"
            " border-radius: 3px;"
            " padding: 0 4px;"
            " background: palette(base);"
            " color: palette(text);"
            " }"
        )
        item.setText(0, "")
        self.model_tree.setItemWidget(item, 0, editor)
        self._active_rename = (item, original_text, item_type)
        editor.selectAll()
        editor.setFocus()
        editor.editingFinished.connect(lambda: self._finish_inline_rename())
        editor.focus_lost.connect(lambda: self._finish_inline_rename())

    def _finish_inline_rename(self) -> None:
        active = self._active_rename
        self._active_rename = None
        if active is None:
            return

        item, original_text, item_type = active
        editor = self.model_tree.itemWidget(item, 0)
        new_text = original_text
        if isinstance(editor, QLineEdit):
            candidate = editor.text().strip()
            if candidate != "":
                new_text = candidate
        self.model_tree.removeItemWidget(item, 0)
        item.setText(0, original_text)

        if new_text == original_text:
            return

        if item_type[0] == "model":
            model_id = int(item_type[1])
            try:
                self.session.rename_model(model_id, new_text)
            except ValueError as exc:
                self._show_error("Rename failed", exc)
                self.refresh(select_model_id=model_id)
                return
            item.setText(0, new_text)
            if self.visualization_window is not None:
                self.visualization_window.refresh()
            return

        model_id = int(item_type[1])
        component_id = int(item_type[2])
        try:
            self.session.rename_component(model_id, component_id, new_text)
        except ValueError as exc:
            self._show_error("Component rename failed", exc)
            self.refresh(select_model_id=model_id)
            return
        item.setText(0, new_text)
        if self.visualization_window is not None:
            self.visualization_window.refresh()

    def _refresh_model_list(self, *, select_model_id: Optional[int] = None):
        expanded_components = self._expanded_component_ids()
        active_selected_model_id = select_model_id
        if active_selected_model_id is None:
            active_selected_model_id = self._selected_model_id()
        if active_selected_model_id is None and self.session.models:
            active_selected_model_id = self.session.models[0].id

        self._updating_tree = True
        blocker = QSignalBlocker(self.model_tree)
        self.model_tree.clear()

        for instance in self.session.models:
            model_color = self._model_color(instance)
            model_item = QTreeWidgetItem(
                [
                    instance.display_name,
                    "shown" if instance.visible else "hidden",
                    self._interval_summary(instance),
                    self._status_text(instance),
                ]
            )
            model_item.setData(0, Qt.ItemDataRole.UserRole, ("model", instance.id))
            model_item.setFlags(
                model_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            model_item.setCheckState(0, Qt.CheckState.Checked if instance.visible else Qt.CheckState.Unchecked)
            model_item.setIcon(0, self._color_icon(model_color))
            self._apply_model_item_styling(model_item, model_color)
            self.model_tree.addTopLevelItem(model_item)

            fit_quality_item = QTreeWidgetItem(
                [
                    "Fit Quality",
                    self._fit_quality_summary(instance),
                    "show band",
                    self._fit_quality_status(instance),
                ]
            )
            fit_quality_item.setData(0, Qt.ItemDataRole.UserRole, ("fit_quality", instance.id))
            fit_quality_item.setFlags(
                fit_quality_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            fit_quality_item.setCheckState(
                2,
                Qt.CheckState.Checked if instance.show_1sigma_band else Qt.CheckState.Unchecked,
            )
            self._apply_model_item_styling(fit_quality_item, model_color, child=True)
            model_item.addChild(fit_quality_item)

            interval_item = QTreeWidgetItem(
                [
                    "Interval",
                    self._interval_summary(instance),
                    "",
                    self._interval_render_status(
                        instance,
                        selected_model_id=active_selected_model_id,
                    ),
                ]
            )
            interval_item.setData(0, Qt.ItemDataRole.UserRole, ("interval", instance.id))
            interval_item.setFlags(
                interval_item.flags()
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            self._apply_model_item_styling(interval_item, model_color, child=True)
            model_item.addChild(interval_item)
            self.model_tree.setItemWidget(
                interval_item,
                2,
                self._create_interval_mode_combo(instance.id, instance.interval_display_mode),
            )

            composition_item = QTreeWidgetItem(
                [
                    "Composition",
                    "",
                    f"{len(instance.components)} component(s)",
                    self._composition_summary(instance),
                ]
            )
            composition_item.setData(0, Qt.ItemDataRole.UserRole, ("composition", instance.id))
            self._apply_model_item_styling(composition_item, model_color, child=True)
            model_item.addChild(composition_item)

            for component in instance.components:
                component_item = QTreeWidgetItem(
                    [
                        component.display_name,
                        "unresolved" if component.load_error is not None else ("enabled" if component.enabled else "disabled"),
                        component.saved_model_type_name
                        or getattr(component.model_type, "__name__", str(component.model_type)),
                        self._component_status_summary(instance, component),
                    ]
                )
                component_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ("component", instance.id, component.id),
                )
                component_flags = (
                    component_item.flags()
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                if component.model_type is not None:
                    component_flags |= Qt.ItemFlag.ItemIsUserCheckable
                component_item.setFlags(component_flags)
                component_item.setCheckState(0, Qt.CheckState.Checked if component.enabled else Qt.CheckState.Unchecked)
                self._apply_model_item_styling(component_item, model_color, child=True)
                composition_item.addChild(component_item)
                for param_item, param_name in self._build_component_param_items(instance, component, model_color):
                    component_item.addChild(param_item)
                    self.model_tree.setItemWidget(
                        param_item,
                        1,
                        self._create_component_param_state_combo(instance.id, component.id, param_name),
                    )
                    self.model_tree.setItemWidget(
                        param_item,
                        2,
                        self._create_component_param_value_widget(instance, component, param_name),
                    )
                component_item.setExpanded((instance.id, component.id) in expanded_components)

            model_item.setExpanded(True)
            composition_item.setExpanded(True)
            if select_model_id is not None and instance.id == select_model_id:
                self.model_tree.setCurrentItem(model_item)

        for column in range(self.model_tree.columnCount()):
            self.model_tree.resizeColumnToContents(column)

        self._updating_tree = False
        del blocker
        if self.model_tree.currentItem() is None and self.model_tree.topLevelItemCount() > 0:
            self.model_tree.setCurrentItem(self.model_tree.topLevelItem(0))
        self._handle_selection_changed()

    def _selected_model_id(self) -> Optional[int]:
        item = self.model_tree.currentItem()
        if item is None:
            return None
        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        if item_type is None:
            return None
        if item_type[0] in {"model", "fit_quality", "interval", "composition"}:
            return item_type[1]
        if item_type[0] == "component":
            return item_type[1]
        if item_type[0] == "component_param":
            return item_type[1]
        return None

    def _selected_component_id(self) -> Optional[int]:
        item = self.model_tree.currentItem()
        if item is None:
            return None
        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        if item_type is None or item_type[0] not in {"component", "component_param"}:
            return None
        return int(item_type[2])

    def _selected_model(self) -> SessionModel | None:
        model_id = self._selected_model_id()
        if model_id is None:
            return None
        return self.session.get_model(model_id)

    def _selected_component(self) -> CompositionComponent | None:
        item = self.model_tree.currentItem()
        if item is None:
            return None
        item_type = item.data(0, Qt.ItemDataRole.UserRole)
        if item_type is None or item_type[0] not in {"component", "component_param"}:
            return None
        return self.session.get_component(item_type[1], item_type[2])

    def _effective_index_interval(self, instance: SessionModel) -> tuple[int, int]:
        if instance.interval is None:
            return (0, len(self.session.x) - 1)
        if instance.interval_kind == "x":
            x_vals = np.asarray(self.session.x, dtype=float)
            xmin, xmax = sorted((float(instance.interval[0]), float(instance.interval[1])))
            indices = np.flatnonzero((x_vals >= xmin) & (x_vals <= xmax))
            if len(indices) == 0:
                return (0, len(self.session.x) - 1)
            return (int(indices[0]), int(indices[-1]))
        return tuple(sorted((int(instance.interval[0]), int(instance.interval[1]))))

    def _interval_summary(self, instance: SessionModel) -> str:
        if instance.interval is None:
            return "full dataset"
        if instance.interval_kind == "x":
            xmin, xmax = sorted((float(instance.interval[0]), float(instance.interval[1])))
            return f"x=[{xmin:.3g}, {xmax:.3g}]"
        start_idx, end_idx = self._effective_index_interval(instance)
        return f"idx=[{start_idx}, {end_idx}]"

    def _interval_render_status(
        self,
        instance: SessionModel,
        *,
        selected_model_id: int | None,
    ) -> str:
        if instance.interval_display_mode == "off":
            return "hidden"
        if instance.interval_display_mode == "always":
            return "shown"
        if instance.id == selected_model_id:
            return "shown"
        return "hidden"

    def _handle_interval_mode_changed(self, model_id: int, combo: QComboBox) -> None:
        mode = combo.currentData()
        if mode not in {"off", "selected-only", "always"}:
            return
        instance = self.session.get_model(model_id)
        if instance.interval_display_mode == mode:
            return
        self.session.set_interval_display_mode(model_id, mode)
        QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))

    def _update_interval_render_statuses(self, *, selected_model_id: int | None) -> None:
        self._updating_tree = True
        try:
            for row in range(self.model_tree.topLevelItemCount()):
                model_item = self.model_tree.topLevelItem(row)
                model_type = model_item.data(0, Qt.ItemDataRole.UserRole)
                if model_type is None or model_type[0] != "model":
                    continue

                instance = self.session.get_model(model_type[1])
                for child_idx in range(model_item.childCount()):
                    child = model_item.child(child_idx)
                    child_type = child.data(0, Qt.ItemDataRole.UserRole)
                    if child_type is None or child_type[0] != "interval":
                        continue
                    child.setText(
                        3,
                        self._interval_render_status(
                            instance,
                            selected_model_id=selected_model_id,
                        ),
                    )
                    break
        finally:
            self._updating_tree = False

    def _fit_quality_summary(self, instance: SessionModel) -> str:
        if not instance.components:
            return "empty"
        if instance.load_warning is not None:
            return "load warning"
        if instance.last_error is not None:
            return "fit failed"
        if instance.result is None:
            return "not fitted"
        return format_fit_quality(instance.result, decimals=3, latex=False)

    def _fit_quality_status(self, instance: SessionModel) -> str:
        if instance.load_warning is not None:
            return "warning"
        if instance.last_error is not None:
            return "failed"
        if instance.result is None:
            return "pending"
        return "ok"

    def _composition_summary(self, instance: SessionModel) -> str:
        if not instance.components:
            return "empty"
        enabled_count = sum(component.enabled for component in instance.components)
        return f"{enabled_count}/{len(instance.components)} active"

    def _component_status_summary(
        self,
        instance: SessionModel,
        component: CompositionComponent,
    ) -> str:
        if component.load_error is not None:
            return component.load_error

        fixed_count = self._component_fixed_count(instance, component)
        if fixed_count == 1:
            return "1 fixed"
        return f"{fixed_count} fixed"

    def _component_fixed_count(
        self,
        instance: SessionModel,
        component: CompositionComponent,
    ) -> int:
        return sum(
            1
            for param_name in self._component_param_names(component)
            if self._component_fixed_value(instance, component, param_name) is not None
        )

    def _build_component_param_items(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        model_color: QColor,
    ) -> list[tuple[QTreeWidgetItem, str]]:
        items: list[tuple[QTreeWidgetItem, str]] = []
        for param_name in self._component_param_names(component):
            state_text, value_text, status_text = self._component_param_display(
                instance,
                component,
                param_name,
            )
            item = QTreeWidgetItem([param_name, "", "", status_text])
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                ("component_param", instance.id, component.id, param_name),
            )
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            self._apply_model_item_styling(item, model_color, child=True)
            items.append((item, param_name))
        return items

    def _component_param_names(self, component: CompositionComponent) -> list[str]:
        if component.model_type is None:
            return []
        return list(self.session._component_param_names(component.model_type))

    def _component_param_display(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> tuple[str, str, str]:
        fixed_value = self._component_fixed_value(instance, component, param_name)
        if fixed_value is not None:
            return ("fixed", f"{fixed_value:.6g}", "fixed")

        fitted_value = self._component_fitted_value(instance, component, param_name)
        if fitted_value is not None:
            return ("free", str(fitted_value), "fit")

        initial_value = self._component_initial_guess_value(instance, component, param_name)
        if initial_value is not None:
            return ("free", f"{initial_value:.6g}", "init")

        return ("free", "auto", "auto")

    def _component_fitted_value(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ):
        if instance.result is None or component.model_type is None or not component.enabled:
            return None

        active_components = [candidate for candidate in instance.components if candidate.enabled]
        try:
            component_number = active_components.index(component) + 1
        except ValueError:
            return None

        try:
            resolved_names = dict(
                self.session._resolve_component_result_param_names(
                    instance,
                    component=component,
                    component_number=component_number,
                    param_names=self._component_param_names(component),
                )
            )
        except KeyError:
            return None

        resolved_name = resolved_names.get(param_name)
        if resolved_name is None:
            return None
        return instance.result.params.get(resolved_name)

    def _component_initial_guess_value(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> float | None:
        storage_name = self._component_storage_param_name(
            instance,
            component,
            param_name,
            source=instance.initial_guess,
        )
        if storage_name is None or instance.initial_guess is None:
            return None
        return instance.initial_guess.get(storage_name)

    def _component_fixed_value(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> float | None:
        storage_name = self._component_storage_param_name(
            instance,
            component,
            param_name,
            source=instance.fixed_params,
        )
        if storage_name is None or instance.fixed_params is None:
            return None
        return instance.fixed_params.get(storage_name)

    def _component_storage_param_name(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
        *,
        source: dict[str, float] | None,
    ) -> str | None:
        if source is None or component.model_type is None:
            return None

        active_components = [candidate for candidate in instance.components if candidate.enabled]
        try:
            component_number = active_components.index(component) + 1
        except ValueError:
            component_number = None

        if component_number is not None:
            suffixed_name = f"{param_name}_{component_number}"
            if suffixed_name in source:
                return suffixed_name

        if len(active_components) == 1 and param_name in source:
            return param_name

        return None

    def _create_component_param_state_combo(
        self,
        model_id: int,
        component_id: int,
        param_name: str,
    ) -> QComboBox:
        combo = AutoPopupComboBox(self.model_tree)
        combo.addItem("free", userData="free")
        combo.addItem("fixed", userData="fixed")
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        combo.setStyleSheet(
            "QComboBox {"
            " border: 1px solid palette(mid);"
            " border-radius: 3px;"
            " padding: 0 12px 0 3px;"
            " margin: 0px;"
            " min-height: 18px;"
            " background: transparent;"
            " color: palette(text);"
            " }"
            "QComboBox::drop-down {"
            " border: none;"
            " width: 10px;"
            " }"
            "QComboBox QAbstractItemView {"
            " background: palette(base);"
            " color: palette(text);"
            " selection-background-color: palette(highlight);"
            " selection-color: palette(highlighted-text);"
            " }"
        )

        instance = self.session.get_model(model_id)
        component = self.session.get_component(model_id, component_id)
        is_fixed = self._component_fixed_value(instance, component, param_name) is not None
        combo.setCurrentIndex(1 if is_fixed else 0)
        combo.activated.connect(
            lambda _idx, model_id=model_id, component_id=component_id, param_name=param_name, combo=combo:
            QTimer.singleShot(
                0,
                lambda model_id=model_id, component_id=component_id, param_name=param_name, combo=combo:
                self._handle_component_param_state_changed(model_id, component_id, param_name, combo)
            )
        )
        return combo

    def _create_component_param_value_widget(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> QWidget:
        fixed_value = self._component_fixed_value(instance, component, param_name)
        if fixed_value is None:
            label = QLabel(self._component_param_display(instance, component, param_name)[1])
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            label.setMargin(4)
            return label

        spin = ClampedFloatSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        spin.setKeyboardTracking(False)
        spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spin.setFrame(False)
        spin.setStyleSheet(
            "QDoubleSpinBox {"
            " border: 1px solid palette(mid);"
            " border-radius: 3px;"
            " padding: 0 4px;"
            " background: transparent;"
            " }"
        )
        spin.setValue(float(fixed_value))
        spin.valueChanged.connect(
            lambda value, model_id=instance.id, component_id=component.id, param_name=param_name:
            self._handle_component_param_value_changed(model_id, component_id, param_name, value)
        )
        return spin

    def _handle_component_param_state_changed(
        self,
        model_id: int,
        component_id: int,
        param_name: str,
        combo: QComboBox,
    ) -> None:
        mode = combo.currentData()
        instance = self.session.get_model(model_id)
        component = self.session.get_component(model_id, component_id)
        currently_fixed = self._component_fixed_value(instance, component, param_name) is not None

        if mode == "fixed" and not currently_fixed:
            value = self._default_fixed_value(instance, component, param_name)
            storage_name = self._component_storage_param_name_for_mode(instance, component, param_name)
            self.session.set_fixed_param_value(model_id, storage_name, value)
            QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))
            return

        if mode == "free" and currently_fixed:
            storage_name = self._component_storage_param_name_for_mode(instance, component, param_name)
            self.session.clear_fixed_param(model_id, storage_name)
            QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))

    def _handle_component_param_value_changed(
        self,
        model_id: int,
        component_id: int,
        param_name: str,
        value: float,
    ) -> None:
        instance = self.session.get_model(model_id)
        component = self.session.get_component(model_id, component_id)
        if self._component_fixed_value(instance, component, param_name) is None:
            return
        storage_name = self._component_storage_param_name_for_mode(instance, component, param_name)
        self.session.set_fixed_param_value(model_id, storage_name, value)
        QTimer.singleShot(0, lambda model_id=model_id: self.refresh(select_model_id=model_id))

    def _component_storage_param_name_for_mode(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> str:
        existing_name = self._component_storage_param_name(
            instance,
            component,
            param_name,
            source=instance.fixed_params,
        )
        if existing_name is not None:
            return existing_name

        active_components = [candidate for candidate in instance.components if candidate.enabled]
        if len(active_components) == 1:
            return param_name
        return f"{param_name}_{active_components.index(component) + 1}"

    def _default_fixed_value(
        self,
        instance: SessionModel,
        component: CompositionComponent,
        param_name: str,
    ) -> float:
        fitted_value = self._component_fitted_value(instance, component, param_name)
        if fitted_value is not None:
            return float(fitted_value.value if hasattr(fitted_value, "value") else fitted_value)
        initial_value = self._component_initial_guess_value(instance, component, param_name)
        if initial_value is not None:
            return float(initial_value)
        return 0.0

    def _status_text(self, instance: SessionModel) -> str:
        if not instance.components:
            return "empty"
        if instance.load_warning is not None:
            return "warning"
        if instance.last_error is not None:
            return "failed"
        if instance.result is not None:
            return "fitted"
        return "not fitted"

    def _show_error(self, title: str, exc: Exception):
        QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")

    def _model_color(self, instance: SessionModel) -> QColor:
        return pg.intColor(instance.color_index, hues=max(1, len(self.session.models)))

    def _apply_model_item_styling(self, item: QTreeWidgetItem, color: QColor, *, child: bool = False):
        background = QColor(color)
        background.setAlpha(18 if child else 28)

        for column in range(self.model_tree.columnCount()):
            item.setBackground(column, QBrush(background))

    def _color_icon(self, color: QColor) -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.GlobalColor.transparent)
        tint = QColor(color)
        tint.setAlpha(230)
        painter = QPainter(pixmap)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tint)
        painter.drawRect(2, 2, 8, 8)
        painter.end()
        return QIcon(pixmap)

    def _move_selected(self, direction: Literal[-1, 1]):
        selected_model = self._selected_model()
        if selected_model is None:
            return

        selected_component = self._selected_component()
        if selected_component is not None:
            self.session.move_component(selected_model.id, selected_component.id, direction)
            self.refresh(select_model_id=selected_model.id)
            self._select_component(selected_model.id, selected_component.id)
            return

        self.session.move_model(selected_model.id, direction)
        self.refresh(select_model_id=selected_model.id)

    def _select_component(self, model_id: int, component_id: int):
        for i in range(self.model_tree.topLevelItemCount()):
            model_item = self.model_tree.topLevelItem(i)
            item_type = model_item.data(0, Qt.ItemDataRole.UserRole)
            if item_type is None or item_type[0] != "model" or item_type[1] != model_id:
                continue
            for child_idx in range(model_item.childCount()):
                child = model_item.child(child_idx)
                child_type = child.data(0, Qt.ItemDataRole.UserRole)
                if child_type is None or child_type[0] != "composition":
                    continue
                for component_idx in range(child.childCount()):
                    component_item = child.child(component_idx)
                    component_type = component_item.data(0, Qt.ItemDataRole.UserRole)
                    if (
                        component_type is not None
                        and component_type[0] == "component"
                        and component_type[1] == model_id
                        and component_type[2] == component_id
                    ):
                        self.model_tree.setCurrentItem(component_item)
                        return

    def select_model(self, model_id: int) -> None:
        for i in range(self.model_tree.topLevelItemCount()):
            model_item = self.model_tree.topLevelItem(i)
            item_type = model_item.data(0, Qt.ItemDataRole.UserRole)
            if item_type is None or item_type[0] != "model" or item_type[1] != model_id:
                continue
            self.model_tree.setCurrentItem(model_item)
            return

    def _selected_interval_kind(self) -> str:
        return str(self.interval_kind_selector.currentData())

    def _sync_interval_input_visibility(self, interval_kind: str):
        use_index = interval_kind == "index"
        self.interval_start_spin.setVisible(use_index)
        self.interval_end_spin.setVisible(use_index)
        self.x_interval_start_spin.setVisible(not use_index)
        self.x_interval_end_spin.setVisible(not use_index)
        for widget, visible in (
            (self.interval_start_spin, use_index),
            (self.interval_end_spin, use_index),
            (self.x_interval_start_spin, not use_index),
            (self.x_interval_end_spin, not use_index),
        ):
            label = self.interval_layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)

    def _full_x_interval(self) -> tuple[float, float]:
        x_values = np.asarray(self.session.x, dtype=float)
        if len(x_values) == 0:
            return (0.0, 0.0)
        return (float(np.min(x_values)), float(np.max(x_values)))

    def _convert_interval_to_x(self, instance: SessionModel) -> tuple[float, float]:
        if instance.interval is None:
            return self._full_x_interval()

        current_kind = self.session._resolve_interval_kind(instance)
        if current_kind == "x":
            start_x, end_x = sorted((float(instance.interval[0]), float(instance.interval[1])))
            return (start_x, end_x)

        start_idx, end_idx = sorted((int(instance.interval[0]), int(instance.interval[1])))
        x_values = np.asarray(self.session.x, dtype=float)
        start_idx = max(0, min(start_idx, len(x_values) - 1))
        end_idx = max(0, min(end_idx, len(x_values) - 1))
        return (float(x_values[start_idx]), float(x_values[end_idx]))

    def _convert_interval_to_index(self, instance: SessionModel) -> tuple[int, int]:
        if instance.interval is None:
            return (0, max(0, len(self.session.x) - 1))

        current_kind = self.session._resolve_interval_kind(instance)
        if current_kind == "index":
            start_idx, end_idx = sorted((int(instance.interval[0]), int(instance.interval[1])))
            max_index = max(0, len(self.session.x) - 1)
            return (
                max(0, min(start_idx, max_index)),
                max(0, min(end_idx, max_index)),
            )

        x_values = np.asarray(self.session.x, dtype=float)
        if len(x_values) == 0:
            return (0, 0)

        start_x, end_x = sorted((float(instance.interval[0]), float(instance.interval[1])))
        start_idx = int(np.argmin(np.abs(x_values - start_x)))
        end_idx = int(np.argmin(np.abs(x_values - end_x)))
        return tuple(sorted((start_idx, end_idx)))

    def _schedule_autofit(self, model_id: int):
        self._pending_autofit_model_id = model_id
        self._autofit_timer.start()

    def _run_pending_autofit(self):
        model_id = self._pending_autofit_model_id
        self._pending_autofit_model_id = None
        if model_id is None:
            return
        try:
            self.session.fit_model(model_id)
        except Exception:
            pass
        self.refresh(select_model_id=model_id)

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        if a0 is not None and a0.key() == Qt.Key.Key_Q:
            self.close_all_windows()
            return
        super().keyPressEvent(a0)

    def closeEvent(self, a0) -> None:
        self.closed.emit()
        if self.visualization_window is not None and self.visualization_window.isVisible() and not self._is_closing_all:
            self.visualization_window.close()
        super().closeEvent(a0)
