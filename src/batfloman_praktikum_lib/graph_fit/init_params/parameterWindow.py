from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .parameterSlider import ParameterSlider
from .render_parts import RenderPart


@dataclass(frozen=True)
class ParameterGroup:
    title: str
    params: list[tuple[str, str]]
    render_part_key: str | None = None


def _normalize_param_name(name: str) -> str:
    aliases = {
        "sigma": "s",
    }
    return aliases.get(name, name)


def infer_parameter_groups(
    param_names: list[str],
    model=None,
) -> list[ParameterGroup] | None:
    components = getattr(model, "_components", None)
    if not components:
        return None

    grouped: list[ParameterGroup] = []

    for idx, comp in enumerate(components, start=1):
        comp_param_names = list(comp.get_param_names())
        suffix = f"_{idx}"
        matching_names = [name for name in param_names if name.endswith(suffix)]

        if len(matching_names) != len(comp_param_names):
            return None

        display_names = [name.removesuffix(suffix) for name in matching_names]
        normalized_actual = [_normalize_param_name(name) for name in display_names]
        normalized_expected = [
            _normalize_param_name(name) for name in comp_param_names
        ]

        if normalized_actual != normalized_expected:
            return None

        entries = list(zip(matching_names, display_names))

        grouped.append(
            ParameterGroup(
                title=comp.__name__,
                params=entries,
                render_part_key=f"component_{idx}",
            )
        )

    used_names = {full_name for group in grouped for full_name, _ in group.params}
    if used_names != set(param_names):
        return None

    return grouped


class CollapsibleGroup(QWidget):
    def __init__(
        self,
        title: str,
        render_part_color: str | None = None,
        render_part_key: str | None = None,
        render_part_visible: bool = False,
        render_part_toggle_callback=None,
    ):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        if render_part_color is not None:
            header_layout.addWidget(_make_color_badge(render_part_color))

        self.toggle = QToolButton(text=title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.toggled.connect(self._set_expanded)
        header_layout.addWidget(self.toggle)

        self.controls_widget = None
        self.view_toggle = None
        if render_part_key is not None and render_part_toggle_callback is not None:
            self.controls_widget = QWidget()
            controls_layout = QHBoxLayout(self.controls_widget)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(6)

            controls_layout.addWidget(QLabel("View"))

            self.view_toggle = QCheckBox()
            self.view_toggle.setChecked(render_part_visible)
            self.view_toggle.toggled.connect(
                lambda visible, key=render_part_key: render_part_toggle_callback(key, visible)
            )
            controls_layout.addWidget(self.view_toggle)

            header_layout.addWidget(self.controls_widget)

        header_layout.addStretch(1)

        layout.addWidget(header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 0, 0, 0)
        layout.addWidget(self.content)

    def _set_expanded(self, expanded: bool):
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class ParameterWindow(QWidget):
    graph_win = None
    _copied_group_values: Optional[dict[str, float]] = None
    on_close_callback = None

    def __init__(
        self,
        params: dict[str, float | ParameterSlider],
        update_callback,
        model=None,
        render_parts: Optional[list[RenderPart]] = None,
        render_part_toggle_callback=None,
    ):
        super().__init__()

        self.setWindowTitle("Manual Initial Parameters")
        self.sliders: dict[str, ParameterSlider] = {}
        self._update_callback = update_callback
        self.render_parts = render_parts or []
        self.render_part_toggle_callback = render_part_toggle_callback
        self.render_part_checkboxes: dict[str, QCheckBox] = {}
        self.render_part_defaults = {
            part.key: part.visible_by_default for part in self.render_parts
        }
        self.render_parts_by_key = {part.key: part for part in self.render_parts}

        outer_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        scroll.setWidget(content)

        groups = infer_parameter_groups(list(params.keys()), model=model)
        if groups is None:
            self._build_flat_layout(params)
            if self.render_parts:
                self._build_render_part_layout()
        else:
            self._build_grouped_layout(params, groups)

        self.layout.addStretch(1)

    def _build_flat_layout(self, params: dict[str, float | ParameterSlider]):
        for name, val in params.items():
            slider = self._ensure_slider(name, val)
            self.layout.addWidget(slider)

    def _build_grouped_layout(
        self,
        params: dict[str, float | ParameterSlider],
        groups: list[ParameterGroup],
    ):
        for group in groups:
            group_widget = CollapsibleGroup(
                group.title,
                render_part_color=self._get_render_part_color(group.render_part_key),
                render_part_key=group.render_part_key,
                render_part_visible=self.render_part_defaults.get(group.render_part_key, False),
                render_part_toggle_callback=self.render_part_toggle_callback,
            )
            if group.render_part_key is not None and group_widget.view_toggle is not None:
                self.render_part_checkboxes[group.render_part_key] = group_widget.view_toggle

            for full_name, display_name in group.params:
                slider = self._ensure_slider(full_name, params[full_name], display_name)
                group_widget.content_layout.addWidget(slider)

            self.layout.addWidget(group_widget)

    def _build_render_part_layout(self):
        group_widget = CollapsibleGroup("View")
        for part in self.render_parts:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            row_layout.addWidget(_make_color_badge(part.color))

            checkbox = QCheckBox(part.label)
            checkbox.setChecked(part.visible_by_default)
            self.render_part_checkboxes[part.key] = checkbox
            checkbox.toggled.connect(
                lambda checked, key=part.key: self.render_part_toggle_callback(key, checked)
            )
            row_layout.addWidget(checkbox)
            row_layout.addStretch(1)
            group_widget.content_layout.addWidget(row)

        self.layout.addWidget(group_widget)

    def _ensure_slider(
        self,
        name: str,
        val: float | ParameterSlider,
        display_name: Optional[str] = None,
    ) -> ParameterSlider:
        if isinstance(val, ParameterSlider):
            slider = val
            slider.update_callback = self._update_callback
            if display_name is not None:
                slider.display_name = display_name
                slider.label.setText(display_name)
        else:
            slider = ParameterSlider(
                name=name,
                display_name=display_name,
                initial_value=val,
                vmin=0.5 * val,
                vmax=1.5 * val,
                update_callback=self._update_callback,
            )

        self.sliders[name] = slider
        return slider

    def get_params(self):
        return {k: s.get_value() for k, s in self.sliders.items()}

    def get_fixed_params(self):
        return {
            name: slider.get_value()
            for name, slider in self.sliders.items()
            if slider.is_fixed()
        }

    def set_render_part_visibility(self, key: str, visible: bool):
        checkbox = self.render_part_checkboxes.get(key)
        if checkbox is None:
            return

        previous = checkbox.blockSignals(True)
        checkbox.setChecked(visible)
        checkbox.blockSignals(previous)

    def _get_render_part_color(self, key: str | None) -> str | None:
        if key is None:
            return None
        part = self.render_parts_by_key.get(key)
        if part is None:
            return None
        return part.color

    def keyPressEvent(self, a0) -> None:  # use base name 'a0' to satisfy checker
        if a0.key() == Qt.Key.Key_Q:
            if self.graph_win is not None:
                self.graph_win.close_all_windows()
            else:
                self.close()
        else:
            super().keyPressEvent(a0)

    def closeEvent(self, a0) -> None:
        callback = self.on_close_callback
        super().closeEvent(a0)
        if callable(callback):
            callback()


def _make_color_badge(color: str | None) -> QWidget:
    css_color = _to_css_color(color)
    badge = QLabel()
    badge.setFixedSize(12, 12)
    badge.setStyleSheet(
        f"background-color: {css_color}; "
        "border: 1px solid palette(mid);"
    )
    return badge


def _to_css_color(color: str | None) -> str:
    if color is None:
        return "transparent"

    aliases = {
        "b": "#1f77b4",
        "m": "#d627a5",
        "c": "#17becf",
        "y": "#bcbd22",
        "w": "#ffffff",
    }
    return aliases.get(color, color)
