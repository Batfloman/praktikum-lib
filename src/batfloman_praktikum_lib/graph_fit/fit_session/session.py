from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .analysis import FitAnalysis
from ..fitResult import FIT_METHODS, FitResult
from ..init_params import ManualFitSetup, manual_fit_setup

IntervalKind = Literal["index", "x"]


@dataclass
class CompositionComponent:
    id: str
    model_type: Any
    enabled: bool = True
    name: Optional[str] = None

    @property
    def display_name(self) -> str:
        if self.name is not None:
            return self.name
        return getattr(self.model_type, "__name__", str(self.model_type))


@dataclass
class SessionModel:
    id: str
    interval: tuple[float, float] | tuple[int, int] | None = None
    interval_kind: IntervalKind | None = None
    name: Optional[str] = None
    visible: bool = True
    cache_path: Optional[str] = None
    excluded_indices: tuple[int, ...] = ()
    components: list[CompositionComponent] = field(default_factory=list)
    initial_guess: dict[str, float] | None = None
    setup: ManualFitSetup | None = None
    result: FitResult | None = None
    last_error: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.id

    def build_model(self):
        active_components = [component.model_type for component in self.components if component.enabled]
        if not active_components:
            return None

        model = active_components[0]
        for component in active_components[1:]:
            model = model + component
        return model


ModelInstance = SessionModel


class FitSession:
    def __init__(
        self,
        x,
        y,
        *,
        xerr=None,
        yerr=None,
        cache_dir: str | Path = "fit_session_cache",
    ):
        self.original_x = np.asarray(x, dtype=object)
        self.original_y = np.asarray(y, dtype=object)
        self.original_xerr = None if xerr is None else np.asarray(xerr, dtype=object)
        self.original_yerr = None if yerr is None else np.asarray(yerr, dtype=object)
        self.x = np.asarray(x, dtype=object)
        self.y = np.asarray(y, dtype=object)
        self.xerr = None if xerr is None else np.asarray(xerr, dtype=object)
        self.yerr = None if yerr is None else np.asarray(yerr, dtype=object)
        self.cache_dir = Path(cache_dir)
        self.state_path = self.cache_dir / "session.json"
        self.models: list[SessionModel] = []
        self._next_model_id = 1
        self.load_state()

    def add_model(
        self,
        model=None,
        *,
        interval: tuple[float, float] | tuple[int, int] | None = None,
        interval_kind: IntervalKind | None = None,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        cache_path: Optional[str] = None,
        components: Optional[list[Any]] = None,
    ) -> str:
        resolved_id = model_id or f"model_{self._next_model_id}"
        if any(instance.id == resolved_id for instance in self.models):
            raise ValueError(f"Duplicate model id: '{resolved_id}'.")

        self._next_model_id += 1
        session_model = SessionModel(
            id=resolved_id,
            interval=interval,
            interval_kind=interval_kind,
            name=name or getattr(model, "__name__", None),
            visible=visible,
            cache_path=cache_path,
        )
        self.models.append(session_model)

        initial_components = list(components or [])
        if model is not None:
            initial_components.insert(0, model)
        for component in initial_components:
            self.add_component(session_model.id, component)
        self.save_state()
        return session_model.id

    def get_model(self, model_id: str) -> SessionModel:
        for instance in self.models:
            if instance.id == model_id:
                return instance
        raise KeyError(f"Unknown model id: '{model_id}'.")

    def set_visible(self, model_id: str, visible: bool) -> None:
        self.get_model(model_id).visible = visible
        self.save_state()

    def remove_model(self, model_id: str) -> SessionModel:
        instance = self.get_model(model_id)
        self.models = [model for model in self.models if model.id != model_id]
        self.save_state()
        return instance

    def add_component(
        self,
        model_id: str,
        model_type,
        *,
        component_id: Optional[str] = None,
        enabled: bool = True,
        name: Optional[str] = None,
    ) -> str:
        session_model = self.get_model(model_id)
        resolved_id = component_id or f"{model_id}_component_{len(session_model.components) + 1}"
        if any(component.id == resolved_id for component in session_model.components):
            raise ValueError(f"Duplicate component id '{resolved_id}' in model '{model_id}'.")

        session_model.components.append(
            CompositionComponent(
                id=resolved_id,
                model_type=model_type,
                enabled=enabled,
                name=name,
            )
        )
        self.invalidate_model(model_id)
        self.save_state()
        return resolved_id

    def remove_component(self, model_id: str, component_id: str) -> CompositionComponent:
        session_model = self.get_model(model_id)
        for component in session_model.components:
            if component.id == component_id:
                session_model.components = [
                    candidate
                    for candidate in session_model.components
                    if candidate.id != component_id
                ]
                self.invalidate_model(model_id)
                self.save_state()
                return component
        raise KeyError(f"Unknown component id '{component_id}' in model '{model_id}'.")

    def get_component(self, model_id: str, component_id: str) -> CompositionComponent:
        session_model = self.get_model(model_id)
        for component in session_model.components:
            if component.id == component_id:
                return component
        raise KeyError(f"Unknown component id '{component_id}' in model '{model_id}'.")

    def set_component_enabled(self, model_id: str, component_id: str, enabled: bool) -> None:
        component = self.get_component(model_id, component_id)
        component.enabled = enabled
        self.invalidate_model(model_id)
        self.save_state()

    def invalidate_model(self, model_id: str) -> None:
        session_model = self.get_model(model_id)
        session_model.setup = None
        session_model.result = None
        session_model.last_error = None

    def set_interval(
        self,
        model_id: str,
        interval: tuple[float, float] | tuple[int, int] | None,
        *,
        interval_kind: IntervalKind | None = None,
    ) -> None:
        session_model = self.get_model(model_id)
        session_model.interval = interval
        if interval_kind is not None:
            session_model.interval_kind = interval_kind
        self.invalidate_model(model_id)
        self.save_state()

    def set_excluded_indices(
        self,
        model_id: str,
        excluded_indices: tuple[int, ...] | list[int] | set[int],
    ) -> None:
        session_model = self.get_model(model_id)
        valid_indices = sorted(
            {
                int(idx)
                for idx in excluded_indices
                if 0 <= int(idx) < len(self.x)
            }
        )
        session_model.excluded_indices = tuple(valid_indices)
        self.invalidate_model(model_id)
        self.save_state()

    def get_composed_model(self, model_id: str):
        return self.get_model(model_id).build_model()

    def open_parameter_editor(self, model_id: str, **kwargs) -> ManualFitSetup:
        instance = self.get_model(model_id)
        setup = self._build_setup(
            instance,
            use_cache=False,
            default_values=self._default_values_for(instance),
            **kwargs,
        )
        global_interval, global_excluded = self._map_setup_selection_to_global(instance, setup)

        instance.interval = global_interval
        instance.interval_kind = "index"
        instance.excluded_indices = global_excluded
        instance.initial_guess = None if setup.initial_guess is None else dict(setup.initial_guess)
        instance.setup = ManualFitSetup(
            model=instance.build_model(),
            x=self.x,
            y=self.y,
            xerr=self.xerr,
            yerr=self.yerr,
            initial_guess=instance.initial_guess,
            interval_indices=global_interval,
            excluded_indices=global_excluded,
        )
        instance.result = None
        instance.last_error = None
        self.save_state()
        return instance.setup

    def fit_model(
        self,
        model_id: str,
        *,
        method: Optional[FIT_METHODS] = None,
        **kwargs,
    ) -> FitResult:
        return self.fit(method=method, model_ids=[model_id], **kwargs)[model_id]

    def get_model_data(self, model_id: str):
        return self._select_data(self.get_model(model_id))

    def analyze(
        self,
        model_id: str,
        *,
        auto_fit: bool = True,
        method: Optional[FIT_METHODS] = None,
        **fit_kwargs,
    ) -> FitAnalysis:
        instance = self.get_model(model_id)
        if instance.result is None and auto_fit:
            self.fit_model(model_id, method=method, **fit_kwargs)
        if instance.result is None:
            raise ValueError(f"Model '{instance.display_name}' has no fit result.")
        return FitAnalysis(
            fit_result=instance.result,
            interval=self._resolve_x_interval(instance),
            model_id=instance.id,
            model_name=instance.display_name,
        )

    def fit(
        self,
        *,
        method: Optional[FIT_METHODS] = None,
        model_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, FitResult]:
        results: dict[str, FitResult] = {}
        instances = self._resolve_instances(model_ids)

        for instance in instances:
            try:
                setup = self._build_runtime_setup(instance)
                instance.result = setup.fit(method=method, **kwargs)
                instance.last_error = None
                results[instance.id] = instance.result
            except Exception as exc:
                instance.last_error = f"{type(exc).__name__}: {exc}"
                raise

        return results

    def plot(
        self,
        *,
        plot=None,
        show_data: bool = True,
        highlight_intervals: bool = True,
    ):
        from batfloman_praktikum_lib import graph

        if plot is None:
            plot = graph.create_plot()

        fig, ax = plot

        if show_data:
            graph.scatter(self.x, self.y, plot=plot)

        for idx, instance in enumerate(self.models):
            if not instance.visible:
                continue

            if highlight_intervals and instance.interval is not None:
                x_interval = self._resolve_x_interval(instance)
                ax.axvspan(
                    x_interval[0],
                    x_interval[1],
                    alpha=0.08,
                    color=f"C{idx % 10}",
                    zorder=0,
                )

            if instance.result is not None:
                x_interval = self._resolve_x_interval(instance)
                graph.plot_func(
                    instance.result,
                    plot=plot,
                    interval=x_interval,
                    color=f"C{idx % 10}",
                    label=instance.display_name,
                    zorder=4,
                )

        return plot

    def _resolve_instances(self, model_ids: Optional[list[str]]) -> list[SessionModel]:
        if model_ids is None:
            return list(self.models)
        return [self.get_model(model_id) for model_id in model_ids]

    def _build_runtime_setup(self, instance: SessionModel) -> ManualFitSetup:
        x_subset, y_subset, xerr_subset, yerr_subset, index_subset = self._select_interval_data(instance)
        local_interval = (0, len(index_subset) - 1) if len(index_subset) > 0 else None
        excluded_lookup = set(instance.excluded_indices)
        local_excluded = tuple(
            local_idx
            for local_idx, global_idx in enumerate(index_subset)
            if global_idx in excluded_lookup
        )
        return ManualFitSetup(
            model=instance.build_model(),
            x=x_subset,
            y=y_subset,
            xerr=xerr_subset,
            yerr=yerr_subset,
            initial_guess=self._default_values_for(instance),
            interval_indices=local_interval,
            excluded_indices=local_excluded,
        )

    def _build_setup(self, instance: SessionModel, *, use_cache: bool = False, **kwargs) -> ManualFitSetup:
        composed_model = instance.build_model()
        if composed_model is None:
            raise ValueError(f"Model '{instance.display_name}' has no active composition components.")

        x_subset, y_subset, xerr_subset, yerr_subset, index_subset = self._select_interval_data(instance)
        local_interval = (0, len(index_subset) - 1) if len(index_subset) > 0 else None
        local_excluded = tuple(
            local_idx
            for local_idx, global_idx in enumerate(index_subset)
            if global_idx in set(instance.excluded_indices)
        )
        return manual_fit_setup(
            composed_model,
            x_subset,
            y_subset,
            xerr=xerr_subset,
            yerr=yerr_subset,
            cache_path=None,
            use_cache=use_cache,
            interval_indices=local_interval,
            excluded_indices=local_excluded,
            **kwargs,
        )

    def _select_data(self, instance: SessionModel):
        mask = self._build_selection_mask(instance)
        x_subset = self.x[mask]
        y_subset = self.y[mask]
        xerr_subset = None if self.xerr is None else self.xerr[mask]
        yerr_subset = None if self.yerr is None else self.yerr[mask]
        return x_subset, y_subset, xerr_subset, yerr_subset

    def _select_interval_data(self, instance: SessionModel):
        mask = self._build_interval_mask(instance)
        x_subset = self.x[mask]
        y_subset = self.y[mask]
        xerr_subset = None if self.xerr is None else self.xerr[mask]
        yerr_subset = None if self.yerr is None else self.yerr[mask]
        indices = np.flatnonzero(mask)
        return x_subset, y_subset, xerr_subset, yerr_subset, indices

    def _build_selection_mask(self, instance: SessionModel) -> np.ndarray:
        mask = self._build_interval_mask(instance)
        if not instance.excluded_indices:
            return mask
        excluded = np.asarray(instance.excluded_indices, dtype=int)
        valid_excluded = excluded[(excluded >= 0) & (excluded < len(mask))]
        mask[valid_excluded] = False
        return mask

    def _build_interval_mask(self, instance: SessionModel) -> np.ndarray:
        mask = np.ones(len(self.x), dtype=bool)
        if instance.interval is None:
            return mask

        interval_kind = self._resolve_interval_kind(instance)
        if interval_kind == "index":
            start_idx, end_idx = sorted((int(instance.interval[0]), int(instance.interval[1])))
            if start_idx < 0 or end_idx >= len(self.x):
                raise IndexError(f"Model '{instance.id}' interval index out of bounds.")
            mask = np.zeros(len(self.x), dtype=bool)
            mask[start_idx:end_idx + 1] = True
            return mask

        x_vals = np.asarray(self.x, dtype=float)
        xmin, xmax = sorted((float(instance.interval[0]), float(instance.interval[1])))
        return (x_vals >= xmin) & (x_vals <= xmax)

    def _resolve_interval_kind(self, instance: SessionModel) -> IntervalKind:
        if instance.interval_kind is not None:
            return instance.interval_kind
        if instance.interval is None:
            return "index"
        if all(isinstance(value, (int, np.integer)) for value in instance.interval):
            return "index"
        return "x"

    def _resolve_x_interval(self, instance: SessionModel) -> tuple[float, float]:
        if instance.interval is None:
            x_vals = np.asarray(self.x, dtype=float)
            return (float(np.min(x_vals)), float(np.max(x_vals)))

        interval_kind = self._resolve_interval_kind(instance)
        if interval_kind == "x":
            xmin, xmax = sorted((float(instance.interval[0]), float(instance.interval[1])))
            return xmin, xmax

        start_idx, end_idx = sorted((int(instance.interval[0]), int(instance.interval[1])))
        x_vals = np.asarray(self.x, dtype=float)
        return float(x_vals[start_idx]), float(x_vals[end_idx])

    def _map_setup_selection_to_global(
        self,
        instance: SessionModel,
        setup: ManualFitSetup,
    ) -> tuple[tuple[int, int] | None, tuple[int, ...]]:
        _, _, _, _, base_indices = self._select_interval_data(instance)
        if len(base_indices) == 0:
            return None, ()

        local_interval = setup.interval_indices
        if local_interval is None:
            global_interval = (int(base_indices[0]), int(base_indices[-1]))
        else:
            start_local, end_local = sorted((int(local_interval[0]), int(local_interval[1])))
            start_local = max(0, min(start_local, len(base_indices) - 1))
            end_local = max(0, min(end_local, len(base_indices) - 1))
            global_interval = (int(base_indices[start_local]), int(base_indices[end_local]))

        local_excluded = np.asarray(setup.excluded_indices, dtype=int)
        valid_local_excluded = local_excluded[
            (local_excluded >= 0) & (local_excluded < len(base_indices))
        ]
        global_excluded = tuple(sorted(int(base_indices[idx]) for idx in valid_local_excluded))
        return global_interval, global_excluded

    def _default_values_for(self, instance: SessionModel):
        if instance.initial_guess is not None:
            return dict(instance.initial_guess)
        if instance.setup is not None and instance.setup.initial_guess is not None:
            return dict(instance.setup.initial_guess)
        return None

    def _resolve_model_type(self, model_name: str):
        from .. import models as models_module

        if not hasattr(models_module, model_name):
            raise KeyError(f"Unknown saved model type '{model_name}'.")
        return getattr(models_module, model_name)

    def _serialize_state(self) -> dict:
        return {
            "models": [
                {
                    "id": instance.id,
                    "name": instance.name,
                    "visible": instance.visible,
                    "interval": list(instance.interval) if instance.interval is not None else None,
                    "interval_kind": instance.interval_kind,
                    "excluded_indices": list(instance.excluded_indices),
                    "components": [
                        {
                            "id": component.id,
                            "name": component.name,
                            "enabled": component.enabled,
                            "model_type": getattr(component.model_type, "__name__", str(component.model_type)),
                        }
                        for component in instance.components
                    ],
                    "initial_guess": None if instance.initial_guess is None else dict(instance.initial_guess),
                }
                for instance in self.models
            ]
        }

    def save_state(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._serialize_state(), indent=2))

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text())
        self.models = []
        max_model_number = 0

        for model_data in state.get("models", []):
            model_id = str(model_data["id"])
            interval_data = model_data.get("interval")
            interval = None if interval_data is None else (interval_data[0], interval_data[1])
            instance = SessionModel(
                id=model_id,
                interval=interval,
                interval_kind=model_data.get("interval_kind"),
                name=model_data.get("name"),
                visible=bool(model_data.get("visible", True)),
                excluded_indices=tuple(int(idx) for idx in model_data.get("excluded_indices", [])),
                initial_guess=None,
                components=[
                    CompositionComponent(
                        id=str(component_data["id"]),
                        model_type=self._resolve_model_type(component_data["model_type"]),
                        enabled=bool(component_data.get("enabled", True)),
                        name=component_data.get("name"),
                    )
                    for component_data in model_data.get("components", [])
                ],
            )

            initial_guess = model_data.get("initial_guess")
            if initial_guess is not None:
                instance.initial_guess = {
                    key: float(value)
                    for key, value in initial_guess.items()
                }
                instance.setup = ManualFitSetup(
                    model=instance.build_model(),
                    x=self.x,
                    y=self.y,
                    xerr=self.xerr,
                    yerr=self.yerr,
                    initial_guess=dict(instance.initial_guess),
                    interval_indices=instance.interval if instance.interval_kind == "index" else None,
                    excluded_indices=instance.excluded_indices,
                )

            self.models.append(instance)

            if model_id.startswith("model_"):
                suffix = model_id.removeprefix("model_")
                if suffix.isdigit():
                    max_model_number = max(max_model_number, int(suffix))

        self._next_model_id = max_model_number + 1 if max_model_number > 0 else 1
