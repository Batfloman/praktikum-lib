from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Callable, Mapping
import inspect
import os
from typing import Any, Literal, Optional

import numpy as np

from ...path_managment import ensure_extension
from .analysis import ComponentFitAnalysis, FitAnalysis
from ..fitResult import FIT_METHODS, FitResult
from ..init_params import ManualFitSetup, manual_fit_setup

IntervalKind = Literal["index", "x"]
IntervalDisplayMode = Literal["off", "selected-only", "always"]
type FitSessionModelType = Callable[..., Any] | type[Any]
type AvailableModels = Mapping[str, FitSessionModelType]


@dataclass
class CompositionComponent:
    id: int
    model_type: FitSessionModelType | None
    enabled: bool = True
    name: Optional[str] = None
    registry_key: Optional[str] = None
    saved_model_type_name: Optional[str] = None
    load_error: Optional[str] = None

    @property
    def display_name(self) -> str:
        if self.name is not None:
            return self.name
        if self.model_type is not None:
            return getattr(self.model_type, "__name__", str(self.model_type))
        if self.saved_model_type_name is not None:
            return self.saved_model_type_name
        if self.registry_key is not None:
            return self.registry_key
        return "Unresolved Component"


@dataclass
class SessionModel:
    id: int
    color_index: int = 0
    interval: tuple[float, float] | tuple[int, int] | None = None
    interval_kind: IntervalKind | None = None
    name: Optional[str] = None
    visible: bool = True
    cache_path: Optional[str] = None
    excluded_indices: tuple[int, ...] = ()
    components: list[CompositionComponent] = field(default_factory=list)
    initial_guess: dict[str, float] | None = None
    fixed_params: dict[str, float] | None = None
    setup: ManualFitSetup | None = None
    result: FitResult | None = None
    last_error: str | None = None
    load_warning: str | None = None
    show_1sigma_band: bool = True
    interval_display_mode: IntervalDisplayMode = "selected-only"

    @property
    def display_name(self) -> str:
        return self.name or str(self.id)

    def build_model(self):
        active_components = [
            component.model_type
            for component in self.components
            if component.enabled and component.model_type is not None
        ]
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
        cache_path: str | Path = "fit_session.json",
        available_models: AvailableModels | None = None,
    ):
        self.original_x = np.asarray(x, dtype=object)
        self.original_y = np.asarray(y, dtype=object)
        self.original_xerr = None if xerr is None else np.asarray(xerr, dtype=object)
        self.original_yerr = None if yerr is None else np.asarray(yerr, dtype=object)
        self.x = np.asarray(x, dtype=object)
        self.y = np.asarray(y, dtype=object)
        self.xerr = None if xerr is None else np.asarray(xerr, dtype=object)
        self.yerr = None if yerr is None else np.asarray(yerr, dtype=object)
        self._sort_data_by_x()
        self.cache_path = Path(ensure_extension(os.fspath(cache_path), ".json"))
        self.available_models = self._build_available_models(available_models)
        self.models: list[SessionModel] = []
        self._next_model_id = 1
        self._next_color_index = 0
        self.load_state()

    def _build_available_models(
        self,
        available_models: AvailableModels | None,
    ) -> dict[str, FitSessionModelType]:
        resolved_models: dict[str, FitSessionModelType] = dict(available_models or {})
        for model in list(resolved_models.values()):
            model_name = getattr(model, "__name__", None)
            if model_name is not None and model_name not in resolved_models:
                resolved_models[model_name] = model
        return resolved_models

    def _sort_data_by_x(self) -> None:
        if len(self.x) <= 1:
            return

        order = np.argsort(np.asarray(self.x, dtype=float), kind="stable")
        self.x = self.x[order]
        self.y = self.y[order]
        if self.xerr is not None:
            self.xerr = self.xerr[order]
        if self.yerr is not None:
            self.yerr = self.yerr[order]

    def get_model_by_name(self, model_name: str) -> SessionModel:
        normalized_name = self._normalize_model_name(model_name)
        for instance in self.models:
            if instance.name == normalized_name:
                return instance
        raise KeyError(f"Unknown model name: '{normalized_name}'.")

    def get_named_model(self, model_name: str) -> SessionModel:
        return self.get_model_by_name(model_name)

    def add_model(
        self,
        model: FitSessionModelType | None = None,
        *,
        interval: tuple[float, float] | tuple[int, int] | None = None,
        interval_kind: IntervalKind | None = None,
        model_id: Optional[int] = None,
        name: Optional[str] = None,
        visible: bool = True,
        cache_path: Optional[str] = None,
        components: Optional[list[FitSessionModelType]] = None,
    ) -> int:
        resolved_id = self._next_model_id if model_id is None else int(model_id)
        if any(instance.id == resolved_id for instance in self.models):
            raise ValueError(f"Duplicate model id: '{resolved_id}'.")

        self._next_model_id = max(self._next_model_id, resolved_id + 1)
        resolved_name = self._deduplicate_model_name(name or getattr(model, "__name__", None))
        session_model = SessionModel(
            id=resolved_id,
            color_index=self._next_color_index,
            interval=interval,
            interval_kind=interval_kind,
            name=resolved_name,
            visible=visible,
            cache_path=cache_path,
        )
        self._next_color_index += 1
        self.models.append(session_model)

        initial_components = list(components or [])
        if model is not None:
            initial_components.insert(0, model)
        for component in initial_components:
            self.add_component(session_model.id, component)
        self.save_state()
        return session_model.id

    def get_model(self, model_id: int) -> SessionModel:
        for instance in self.models:
            if instance.id == model_id:
                return instance
        raise KeyError(f"Unknown model id: '{model_id}'.")

    def set_visible(self, model_id: int, visible: bool) -> None:
        self.get_model(model_id).visible = visible
        self.save_state()

    def set_show_1sigma_band(self, model_id: int, enabled: bool) -> None:
        self.get_model(model_id).show_1sigma_band = enabled
        self.save_state()

    def set_interval_display_mode(
        self,
        model_id: int,
        mode: IntervalDisplayMode,
    ) -> None:
        self.get_model(model_id).interval_display_mode = mode
        self.save_state()

    def rename_model(self, model_id: int, new_name: str) -> None:
        instance = self.get_model(model_id)
        normalized_name = self._normalize_model_name(new_name)
        if any(
            other.id != model_id and other.name == normalized_name
            for other in self.models
        ):
            raise ValueError(f"Duplicate model name: '{normalized_name}'.")
        instance.name = normalized_name
        self.save_state()

    def remove_model(self, model_id: int) -> SessionModel:
        instance = self.get_model(model_id)
        self.models = [model for model in self.models if model.id != model_id]
        self.save_state()
        return instance

    def move_model(self, model_id: int, direction: Literal[-1, 1]) -> None:
        current_index = self._find_model_index(model_id)
        new_index = current_index + direction
        if new_index < 0 or new_index >= len(self.models):
            return
        self.models[current_index], self.models[new_index] = (
            self.models[new_index],
            self.models[current_index],
        )
        self.save_state()

    def add_component(
        self,
        model_id: int,
        model_type: FitSessionModelType,
        *,
        component_id: Optional[int] = None,
        enabled: bool = True,
        name: Optional[str] = None,
    ) -> int:
        session_model = self.get_model(model_id)
        resolved_id = (
            max((component.id for component in session_model.components), default=0) + 1
            if component_id is None
            else int(component_id)
        )
        if any(component.id == resolved_id for component in session_model.components):
            raise ValueError(f"Duplicate component id '{resolved_id}' in model '{model_id}'.")

        session_model.components.append(
            CompositionComponent(
                id=resolved_id,
                model_type=model_type,
                enabled=enabled,
                name=self._deduplicate_component_name(
                    session_model,
                    name or getattr(model_type, "__name__", None),
                ),
                registry_key=self._registry_key_for_model_type(model_type),
            )
        )
        self.invalidate_model(model_id)
        self.save_state()
        return resolved_id

    def remove_component(self, model_id: int, component_id: int) -> CompositionComponent:
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

    def get_component(self, model_id: int, component_id: int) -> CompositionComponent:
        session_model = self.get_model(model_id)
        for component in session_model.components:
            if component.id == component_id:
                return component
        raise KeyError(f"Unknown component id '{component_id}' in model '{model_id}'.")

    def set_component_enabled(self, model_id: int, component_id: int, enabled: bool) -> None:
        component = self.get_component(model_id, component_id)
        if enabled and component.model_type is None:
            raise ValueError(
                f"Component '{component.display_name}' cannot be enabled because its saved model type is unavailable."
            )
        component.enabled = enabled
        self.invalidate_model(model_id)
        self.save_state()

    def rename_component(self, model_id: int, component_id: int, new_name: str) -> None:
        session_model = self.get_model(model_id)
        component = self.get_component(model_id, component_id)
        normalized_name = self._normalize_component_name(new_name)
        if any(
            other.id != component_id and other.name == normalized_name
            for other in session_model.components
        ):
            raise ValueError(
                f"Duplicate component name: '{normalized_name}' in model '{session_model.display_name}'."
            )
        component.name = normalized_name
        self.save_state()

    def move_component(
        self,
        model_id: int,
        component_id: int,
        direction: Literal[-1, 1],
    ) -> None:
        session_model = self.get_model(model_id)
        old_components = list(session_model.components)
        current_index = self._find_component_index(session_model, component_id)
        new_index = current_index + direction
        if new_index < 0 or new_index >= len(session_model.components):
            return
        session_model.components[current_index], session_model.components[new_index] = (
            session_model.components[new_index],
            session_model.components[current_index],
        )
        session_model.initial_guess = self._remap_initial_guess_for_component_order(
            session_model,
            old_components=old_components,
        )
        self.invalidate_model(model_id)
        self.save_state()

    def invalidate_model(self, model_id: int) -> None:
        session_model = self.get_model(model_id)
        session_model.setup = None
        session_model.result = None
        session_model.last_error = None

    def set_interval(
        self,
        model_id: int,
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
        model_id: int,
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

    def get_composed_model(self, model_id: int):
        return self.get_model(model_id).build_model()

    def open_parameter_editor(self, model_id: int, **kwargs) -> ManualFitSetup:
        instance = self.get_model(model_id)
        resolved_initial_guess = self._default_values_for(instance)
        if resolved_initial_guess is not None:
            instance.initial_guess = dict(resolved_initial_guess)
        setup = self._build_setup(
            instance,
            use_cache=False,
            default_values=resolved_initial_guess,
            **kwargs,
        )
        global_interval, global_excluded = self._map_setup_selection_to_global(instance, setup)

        instance.interval = global_interval
        instance.interval_kind = "index"
        instance.excluded_indices = global_excluded
        instance.initial_guess = None if setup.initial_guess is None else dict(setup.initial_guess)
        instance.fixed_params = None if setup.fixed_params is None else dict(setup.fixed_params)
        instance.setup = ManualFitSetup(
            model=instance.build_model(),
            x=self.x,
            y=self.y,
            xerr=self.xerr,
            yerr=self.yerr,
            initial_guess=instance.initial_guess,
            fixed_params=instance.fixed_params,
            interval_indices=global_interval,
            excluded_indices=global_excluded,
        )
        try:
            instance.result = instance.setup.fit()
            instance.last_error = None
        except Exception as exc:
            instance.result = None
            instance.last_error = f"{type(exc).__name__}: {exc}"
            self.save_state()
            raise
        self.save_state()
        return instance.setup

    def fit_model(
        self,
        model_id: int,
        *,
        method: Optional[FIT_METHODS] = None,
        **kwargs,
    ) -> FitResult:
        return self.fit(method=method, model_ids=[model_id], **kwargs)[model_id]

    def get_model_data(self, model_id: int):
        return self._select_data(self.get_model(model_id))

    def analyze(
        self,
        model_ref: str | int,
        *,
        auto_fit: bool = True,
        method: Optional[FIT_METHODS] = None,
        **fit_kwargs,
    ) -> FitAnalysis:
        instance = self._resolve_analysis_model(model_ref)
        if instance.result is None and auto_fit:
            self.fit_model(instance.id, method=method, **fit_kwargs)
        if instance.result is None:
            raise ValueError(f"Model '{instance.display_name}' has no fit result.")
        return FitAnalysis(
            fit_result=instance.result,
            interval=self._resolve_x_interval(instance),
            model_id=instance.id,
            model_name=instance.display_name,
            components=self._build_component_analyses(instance),
        )

    def fit(
        self,
        *,
        method: Optional[FIT_METHODS] = None,
        model_ids: Optional[list[int]] = None,
        **kwargs,
    ) -> dict[int, FitResult]:
        results: dict[int, FitResult] = {}
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

    def try_fit_models(
        self,
        *,
        method: Optional[FIT_METHODS] = None,
        model_ids: Optional[list[int]] = None,
        **kwargs,
    ) -> dict[int, FitResult]:
        results: dict[int, FitResult] = {}
        for instance in self._resolve_instances(model_ids):
            try:
                results[instance.id] = self.fit_model(instance.id, method=method, **kwargs)
            except Exception:
                continue
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

    def _resolve_instances(self, model_ids: Optional[list[int]]) -> list[SessionModel]:
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
            fixed_params=None if instance.fixed_params is None else dict(instance.fixed_params),
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
            fixed_params=None if instance.fixed_params is None else dict(instance.fixed_params),
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
        auto_defaults = self._auto_initial_guess_for(instance)

        stored_values: dict[str, float] = {}
        if instance.setup is not None and instance.setup.initial_guess is not None:
            stored_values.update(instance.setup.initial_guess)
        if instance.initial_guess is not None:
            stored_values.update(instance.initial_guess)

        if auto_defaults is None:
            return dict(stored_values) if stored_values else None
        return {**auto_defaults, **stored_values}

    def _auto_initial_guess_for(self, instance: SessionModel) -> dict[str, float] | None:
        composed_model = instance.build_model()
        if composed_model is None:
            return None
        if not hasattr(composed_model, "get_initial_guess") or not hasattr(composed_model, "get_param_names"):
            return None

        x_subset, y_subset, _, _ = self._select_data(instance)
        if len(x_subset) == 0 or len(y_subset) == 0:
            return None

        param_names = list(composed_model.get_param_names())  # type: ignore[call-arg]
        initial_guess = list(composed_model.get_initial_guess(x_subset, y_subset))  # type: ignore[call-arg]
        return {
            param_name: float(value)
            for param_name, value in zip(param_names, initial_guess)
        }

    def _resolve_model_type(self, model_name: str | None):
        if model_name is None:
            raise KeyError("Saved model type is missing.")
        if model_name in self.available_models:
            return self.available_models[model_name]
        from .. import models as models_module

        if not hasattr(models_module, model_name):
            raise KeyError(f"Unknown saved model type '{model_name}'.")
        return getattr(models_module, model_name)

    def _registry_key_for_model_type(self, model_type: FitSessionModelType) -> str | None:
        for registry_key, candidate in self.available_models.items():
            if candidate is model_type and registry_key != getattr(model_type, "__name__", None):
                return registry_key
        return getattr(model_type, "__name__", None)

    def _load_component_from_state(self, component_data: dict[str, Any]) -> CompositionComponent:
        saved_registry_key = component_data.get("registry_key")
        saved_model_type_name = component_data.get("model_type")
        resolution_candidates = [
            candidate
            for candidate in (saved_registry_key, saved_model_type_name)
            if candidate is not None
        ]

        resolved_model_type = None
        load_error = None
        for candidate in resolution_candidates:
            try:
                resolved_model_type = self._resolve_model_type(candidate)
                break
            except KeyError:
                continue

        enabled = bool(component_data.get("enabled", True))
        if resolved_model_type is None:
            primary_name = saved_registry_key or saved_model_type_name or "<missing>"
            load_error = f"Unknown saved model type '{primary_name}'."
            enabled = False

        return CompositionComponent(
            id=int(component_data["id"]),
            model_type=resolved_model_type,
            enabled=enabled,
            name=component_data.get("name"),
            registry_key=saved_registry_key,
            saved_model_type_name=saved_model_type_name,
            load_error=load_error,
        )

    def _resolve_analysis_model(self, model_ref: str | int) -> SessionModel:
        if isinstance(model_ref, (int, np.integer)):
            return self.get_model(int(model_ref))
        return self.get_model_by_name(model_ref)

    def _normalize_model_name(self, model_name: str) -> str:
        normalized_name = str(model_name).strip()
        if normalized_name == "":
            raise ValueError("Model name must not be empty.")
        return normalized_name

    def _deduplicate_model_name(self, model_name: str | None) -> str | None:
        if model_name is None:
            return None
        normalized_name = self._normalize_model_name(model_name)
        if not any(instance.name == normalized_name for instance in self.models):
            return normalized_name

        suffix = 2
        while True:
            candidate = f"{normalized_name} {suffix}"
            if not any(instance.name == candidate for instance in self.models):
                return candidate
            suffix += 1

    def _validate_unique_model_names(self) -> None:
        seen_names: set[str] = set()
        for instance in self.models:
            if instance.name is None:
                continue
            normalized_name = self._normalize_model_name(instance.name)
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate saved model name: '{normalized_name}'.")
            instance.name = normalized_name
            seen_names.add(normalized_name)

    def _normalize_component_name(self, component_name: str) -> str:
        normalized_name = str(component_name).strip()
        if normalized_name == "":
            raise ValueError("Component name must not be empty.")
        return normalized_name

    def _deduplicate_component_name(
        self,
        session_model: SessionModel,
        component_name: str | None,
        *,
        exclude_component_id: int | None = None,
    ) -> str | None:
        if component_name is None:
            return None
        normalized_name = self._normalize_component_name(component_name)
        if not any(
            component.id != exclude_component_id and component.name == normalized_name
            for component in session_model.components
        ):
            return normalized_name

        suffix = 2
        while True:
            candidate = f"{normalized_name} {suffix}"
            if not any(
                component.id != exclude_component_id and component.name == candidate
                for component in session_model.components
            ):
                return candidate
            suffix += 1

    def _validate_unique_component_names(self) -> None:
        for instance in self.models:
            seen_names: set[str] = set()
            for component in instance.components:
                if component.name is None:
                    continue
                normalized_name = self._normalize_component_name(component.name)
                if normalized_name in seen_names:
                    raise ValueError(
                        f"Duplicate saved component name '{normalized_name}' in model '{instance.display_name}'."
                    )
                component.name = normalized_name
                seen_names.add(normalized_name)

    def _serialize_state(self) -> dict:
        return {
            "models": [
                {
                    "id": instance.id,
                    "color_index": instance.color_index,
                    "name": instance.name,
                    "visible": instance.visible,
                    "interval": list(instance.interval) if instance.interval is not None else None,
                    "interval_kind": instance.interval_kind,
                    "excluded_indices": list(instance.excluded_indices),
                    "components": [
                        {
                            "id": component.id,
                            "name": component.name,
                            "registry_key": component.registry_key,
                            "enabled": component.enabled,
                            "model_type": component.saved_model_type_name
                            or getattr(component.model_type, "__name__", str(component.model_type)),
                        }
                        for component in instance.components
                    ],
                    "initial_guess": None if instance.initial_guess is None else dict(instance.initial_guess),
                    "fixed_params": None if instance.fixed_params is None else dict(instance.fixed_params),
                    "show_1sigma_band": instance.show_1sigma_band,
                    "interval_display_mode": instance.interval_display_mode,
                }
                for instance in self.models
            ]
        }

    def save_state(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._serialize_state(), indent=2))

    def load_state(self) -> None:
        if not self.cache_path.exists():
            return
        state = json.loads(self.cache_path.read_text())
        self.models = []
        max_model_number = 0

        for model_data in state.get("models", []):
            model_id = int(model_data["id"])
            interval_data = model_data.get("interval")
            interval = None if interval_data is None else (interval_data[0], interval_data[1])
            instance = SessionModel(
                id=model_id,
                color_index=int(model_data.get("color_index", max_model_number)),
                interval=interval,
                interval_kind=model_data.get("interval_kind"),
                name=model_data.get("name"),
                visible=bool(model_data.get("visible", True)),
                excluded_indices=tuple(int(idx) for idx in model_data.get("excluded_indices", [])),
                show_1sigma_band=bool(model_data.get("show_1sigma_band", True)),
                interval_display_mode=self._load_interval_display_mode(model_data),
                initial_guess=None,
                fixed_params=None,
                components=[
                    self._load_component_from_state(component_data)
                    for component_data in model_data.get("components", [])
                ],
            )
            unresolved_components = [
                component.display_name
                for component in instance.components
                if component.load_error is not None
            ]
            if unresolved_components:
                joined = ", ".join(unresolved_components)
                instance.load_warning = (
                    "Unavailable saved component types were kept disabled: "
                    f"{joined}"
                )

            initial_guess = model_data.get("initial_guess")
            if initial_guess is not None:
                instance.initial_guess = {
                    key: float(value)
                    for key, value in initial_guess.items()
                }
            fixed_params = model_data.get("fixed_params")
            if fixed_params is not None:
                instance.fixed_params = {
                    key: float(value)
                    for key, value in fixed_params.items()
                }
            built_model = instance.build_model()
            if built_model is not None and instance.initial_guess is not None:
                instance.setup = ManualFitSetup(
                    model=built_model,
                    x=self.x,
                    y=self.y,
                    xerr=self.xerr,
                    yerr=self.yerr,
                    initial_guess=dict(instance.initial_guess),
                    fixed_params=None if instance.fixed_params is None else dict(instance.fixed_params),
                    interval_indices=instance.interval if instance.interval_kind == "index" else None,
                    excluded_indices=instance.excluded_indices,
                )

            self.models.append(instance)

            max_model_number = max(max_model_number, model_id)

        self._next_model_id = max_model_number + 1 if max_model_number > 0 else 1
        self._next_color_index = (
            max((instance.color_index for instance in self.models), default=-1) + 1
        )
        self._validate_unique_model_names()
        self._validate_unique_component_names()

    def _load_interval_display_mode(self, model_data: dict[str, Any]) -> IntervalDisplayMode:
        saved_mode = model_data.get("interval_display_mode")
        if saved_mode in {"off", "selected-only", "always"}:
            return saved_mode

        legacy_highlight = model_data.get("highlight_interval")
        if legacy_highlight is False:
            return "off"
        return "selected-only"

    def _find_model_index(self, model_id: int) -> int:
        for idx, instance in enumerate(self.models):
            if instance.id == model_id:
                return idx
        raise KeyError(f"Unknown model id: '{model_id}'.")

    def _find_component_index(self, session_model: SessionModel, component_id: int) -> int:
        for idx, component in enumerate(session_model.components):
            if component.id == component_id:
                return idx
        raise KeyError(f"Unknown component id '{component_id}' in model '{session_model.id}'.")

    def _component_param_names(self, model_type: FitSessionModelType) -> list[str]:
        if hasattr(model_type, "get_param_names"):
            return list(model_type.get_param_names())  # type: ignore[call-arg]
        params = list(inspect.signature(model_type).parameters.keys())[1:]
        return params

    def _active_component_blocks(
        self,
        components: list[CompositionComponent],
    ) -> list[tuple[int, list[str]]]:
        return [
            (component.id, self._component_param_names(component.model_type))
            for component in components
            if component.enabled
        ]

    def _build_component_analyses(
        self,
        instance: SessionModel,
    ) -> list[ComponentFitAnalysis]:
        if instance.result is None or not hasattr(instance.result, "params"):
            return []

        active_components = [component for component in instance.components if component.enabled]
        if any(
            not hasattr(component.model_type, "model") or not hasattr(component.model_type, "get_param_names")
            for component in active_components
        ):
            return []

        component_analyses: list[ComponentFitAnalysis] = []
        for position, component in enumerate(active_components, start=1):
            component_param_names = self._component_param_names(component.model_type)
            resolved_param_names = self._resolve_component_result_param_names(
                instance,
                component=component,
                component_number=position,
                param_names=component_param_names,
            )
            component_params = {
                param_name: instance.result.params[resolved_name]
                for param_name, resolved_name in resolved_param_names
            }
            component_analyses.append(
                ComponentFitAnalysis(
                    component_id=component.id,
                    order=position,
                    name=component.display_name,
                    model_name=getattr(component.model_type, "__name__", str(component.model_type)),
                    model_type=component.model_type,
                    params=self._build_component_param_dataset(component_params),
                    evaluate_func=self._build_component_eval(component.model_type, component_params),
                    evaluate_nominal_func=self._build_component_nominal_eval(component.model_type, component_params),
                    interval=self._resolve_x_interval(instance),
                )
            )
        return component_analyses

    def _resolve_component_result_param_names(
        self,
        instance: SessionModel,
        *,
        component: CompositionComponent,
        component_number: int,
        param_names: list[str],
    ) -> list[tuple[str, str]]:
        assert instance.result is not None
        result_params = instance.result.params
        suffixed_names = [
            (param_name, f"{param_name}_{component_number}")
            for param_name in param_names
        ]
        if all(flat_name in result_params for _, flat_name in suffixed_names):
            return suffixed_names

        if len([candidate for candidate in instance.components if candidate.enabled]) == 1:
            raw_names = [(param_name, param_name) for param_name in param_names]
            if all(raw_name in result_params for _, raw_name in raw_names):
                return raw_names

        missing_names = ", ".join(flat_name for _, flat_name in suffixed_names)
        raise KeyError(
            f"Could not resolve fit result parameters for component '{component.display_name}'. "
            f"Expected names like: {missing_names}"
        )

    def _build_component_param_dataset(self, params: dict[str, Any]):
        from ...structs.dataset import Dataset

        return Dataset(params)

    def _build_component_eval(self, model_type: FitSessionModelType, params: dict[str, Any]):
        ordered_names = list(self._component_param_names(model_type))

        def evaluate(x_val):
            values = [params[name] for name in ordered_names]
            if isinstance(x_val, (list, np.ndarray)):
                return [model_type.model(x, *values) for x in x_val]  # type: ignore[attr-defined]
            return model_type.model(x_val, *values)  # type: ignore[attr-defined]

        return evaluate

    def _build_component_nominal_eval(self, model_type: FitSessionModelType, params: dict[str, Any]):
        ordered_values = [
            float(params[name].value if hasattr(params[name], "value") else params[name])
            for name in self._component_param_names(model_type)
        ]

        def evaluate_nominal(x_val):
            if isinstance(x_val, (list, np.ndarray)):
                return np.asarray([model_type.model(x, *ordered_values) for x in x_val])  # type: ignore[attr-defined]
            return model_type.model(x_val, *ordered_values)  # type: ignore[attr-defined]

        return evaluate_nominal

    def _flat_param_names_for_blocks(
        self,
        blocks: list[tuple[int, list[str]]],
    ) -> list[tuple[int, str, str]]:
        flat_names: list[tuple[int, str, str]] = []
        component_number = 1
        for component_id, param_names in blocks:
            for param_name in param_names:
                flat_names.append((component_id, param_name, f"{param_name}_{component_number}"))
            component_number += 1
        return flat_names

    def _remap_initial_guess_for_component_order(
        self,
        session_model: SessionModel,
        *,
        old_components: list[CompositionComponent],
    ) -> dict[str, float] | None:
        if session_model.initial_guess is None:
            return None

        old_blocks = self._active_component_blocks(old_components)
        new_blocks = self._active_component_blocks(session_model.components)
        old_flat_names = self._flat_param_names_for_blocks(old_blocks)
        new_flat_names = self._flat_param_names_for_blocks(new_blocks)

        component_param_values: dict[tuple[int, str], float] = {}
        for component_id, param_name, flat_name in old_flat_names:
            if flat_name in session_model.initial_guess:
                component_param_values[(component_id, param_name)] = session_model.initial_guess[flat_name]

        remapped_guess = dict(session_model.initial_guess)
        for _, _, flat_name in old_flat_names:
            remapped_guess.pop(flat_name, None)

        for component_id, param_name, flat_name in new_flat_names:
            key = (component_id, param_name)
            if key in component_param_values:
                remapped_guess[flat_name] = component_param_values[key]

        return remapped_guess
