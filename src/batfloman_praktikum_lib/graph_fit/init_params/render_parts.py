from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from typing import Any

from ..models import FitModel


@dataclass(frozen=True)
class RenderPart:
    key: str
    label: str
    evaluator: Callable[[Any, dict[str, float]], Any]
    color: str | None = None
    visible_by_default: bool = False
    group_key: str | None = None


DEFAULT_RENDER_PART_COLORS = ["b", "m", "c", "y", "w"]


def resolve_render_parts(
    model,
    render_parts: Mapping[str, Callable] | list[RenderPart] | None = None,
) -> list[RenderPart]:
    if render_parts is None:
        render_parts = getattr(model, "_render_parts", None)

    if render_parts is not None:
        return _normalize_render_parts(render_parts)

    components = getattr(model, "_components", None)
    if components:
        return _composite_render_parts(components)

    return []


def _normalize_render_parts(
    render_parts: Mapping[str, Callable] | list[RenderPart],
) -> list[RenderPart]:
    if isinstance(render_parts, list):
        return _with_default_colors(render_parts)

    normalized: list[RenderPart] = []
    for idx, (label, func) in enumerate(render_parts.items(), start=1):
        normalized.append(
            RenderPart(
                key=f"part_{idx}",
                label=label,
                evaluator=_wrap_part_callable(func),
            )
        )

    return _with_default_colors(normalized)


def _wrap_part_callable(func: Callable) -> Callable[[Any, dict[str, float]], Any]:
    param_names = list(inspect.signature(func).parameters.keys())[1:]

    def evaluator(x, params: dict[str, float]):
        args = [params[name] for name in param_names]
        return func(x, *args)

    return evaluator


def _composite_render_parts(components: list[type[FitModel]]) -> list[RenderPart]:
    parts: list[RenderPart] = []

    for idx, comp in enumerate(components, start=1):
        component_param_names = list(comp.get_param_names())
        full_param_names = [f"{name}_{idx}" for name in component_param_names]

        def evaluator(x, params: dict[str, float], *, comp=comp, full_param_names=full_param_names):
            args = [params[name] for name in full_param_names]
            return comp.model(x, *args)

        parts.append(
            RenderPart(
                key=f"component_{idx}",
                label=f"{comp.__name__} #{idx}",
                evaluator=evaluator,
                group_key=f"component_{idx}",
            )
        )

    return _with_default_colors(parts)


def _with_default_colors(parts: list[RenderPart]) -> list[RenderPart]:
    normalized: list[RenderPart] = []

    for idx, part in enumerate(parts):
        normalized.append(
            RenderPart(
                key=part.key,
                label=part.label,
                evaluator=part.evaluator,
                color=part.color or DEFAULT_RENDER_PART_COLORS[idx % len(DEFAULT_RENDER_PART_COLORS)],
                visible_by_default=part.visible_by_default,
                group_key=part.group_key,
            )
        )

    return normalized
