from dataclasses import dataclass
from inspect import isclass
import json
from typing import Any, Optional, Union, List, Callable, Type
import numpy as np
from pathlib import Path
import inspect

from PyQt6.QtCore import QObject, QEventLoop, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication
from ...flags import should_skip_popup_sequence
from .parameterWindow import ParameterWindow
from .fitSelectionWindow import FitSelectionWindow
from .graphWindow import GraphWindow
from .parameterSlider import ParameterSlider, save_slider_settings, load_slider_settings
from ..models import FitModel
from ._helper import extract_default_values, get_model_fn
from .order_init_params import order_initial_params
from .render_parts import resolve_render_parts
from ..fitResult import FIT_METHODS, FitResult
from ...structs.measurementBase import MeasurementBase

FIT_SELECTION_CACHE_KEY = "__fit_selection__"


class ManualFitSetupController(QObject):
    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        filepath: Path | None,
        model_info,
        x_data,
        y_data,
        xerr,
        yerr,
        param_win: ParameterWindow,
        graph_win: GraphWindow,
        fit_selection_win: FitSelectionWindow,
    ):
        super().__init__()
        self.filepath = filepath
        self.model_info = model_info
        self.x_data = x_data
        self.y_data = y_data
        self.xerr = xerr
        self.yerr = yerr
        self.param_win = param_win
        self.graph_win = graph_win
        self.fit_selection_win = fit_selection_win
        self.result: "ManualFitSetup | None" = None
        self._finished = False

        self.graph_win.on_close_callback = self._schedule_finalize_check
        self.param_win.on_close_callback = self._schedule_finalize_check
        self.fit_selection_win.on_close_callback = self._schedule_finalize_check

    def _schedule_finalize_check(self):
        QTimer.singleShot(0, self._finalize_if_closed)

    def _finalize_if_closed(self):
        if self._finished:
            return
        if any(
            window.isVisible()
            for window in (self.graph_win, self.param_win, self.fit_selection_win)
        ):
            return

        if self.filepath is not None:
            save_slider_settings(self.filepath, self.param_win.sliders)
            cached = load_slider_settings(self.filepath)
            _save_cached_fit_selection(
                self.filepath,
                cached,
                interval_indices=self.graph_win.get_interval_indices(),
                excluded_indices=self.graph_win.get_excluded_indices(),
            )

        self.result = ManualFitSetup(
            model=self.model_info,
            x=self.x_data,
            y=self.y_data,
            xerr=self.xerr,
            yerr=self.yerr,
            initial_guess=self.param_win.get_params(),
            fixed_params=self.param_win.get_fixed_params(),
            interval_indices=self.graph_win.get_interval_indices(),
            excluded_indices=self.graph_win.get_excluded_indices(),
        )
        self._finished = True
        self.finished.emit(self.result)


@dataclass
class ManualFitSetup:
    model: Union[Callable, Type[FitModel]]
    x: Any
    y: Any
    xerr: Any = None
    yerr: Any = None
    initial_guess: dict[str, float] | None = None
    fixed_params: dict[str, float] | None = None
    interval_indices: tuple[int, int] | None = None
    excluded_indices: tuple[int, ...] = ()

    def fit(
        self,
        *,
        method: Optional[FIT_METHODS] = None,
        xerr=None,
        yerr=None,
    ) -> FitResult:
        from ..least_squares import generic_fit as least_squares_fit
        from ..orthogonal_distance import (
            generic_fit as orthogonal_distance_regression_fit,
        )

        bound_xerr = self.xerr if xerr is None else xerr
        bound_yerr = self.yerr if yerr is None else yerr

        x_values, y_values, x_errors, y_errors = _select_fit_data(
            self.x,
            self.y,
            xerr=bound_xerr,
            yerr=bound_yerr,
            interval_indices=self.interval_indices,
            excluded_indices=self.excluded_indices,
        )

        if isclass(self.model) and issubclass(self.model, FitModel):
            return self.model.fit(
                x_values,
                y_values,
                xerr=x_errors,
                yerr=y_errors,
                initial_guess=self.initial_guess,
                fixed_params=self.fixed_params,
                method=method,
            )

        if method == "least squares":
            return least_squares_fit(
                self.model,
                x_values,
                y_values,
                y_errors,
                initial_guess=self.initial_guess,
                fixed_params=self.fixed_params,
                ignore_warning_x_errors=True,
            )

        if method == "ODR" or _should_use_odr(x_values, x_errors):
            return orthogonal_distance_regression_fit(
                self.model,
                x_values,
                y_values,
                x_err=x_errors,
                y_err=y_errors,
                initial_guess=self.initial_guess,
                fixed_params=self.fixed_params,
            )

        return least_squares_fit(
            self.model,
            x_values,
            y_values,
            y_errors,
            initial_guess=self.initial_guess,
            fixed_params=self.fixed_params,
        )


def _filter_nan_data(x_data, y_data, *, warn_filter_nan: bool):
    x_arr = np.array(x_data, dtype=float)
    y_arr = np.array(y_data, dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)

    from batfloman_praktikum_lib.graph.plotNScatter import filter_nan_values
    x_clean, y_clean = filter_nan_values(
        np.asarray(x_data, dtype=object),
        np.asarray(y_data, dtype=object),
        warn_filter_nan=warn_filter_nan,
    )
    return x_clean, y_clean, mask, int(np.count_nonzero(~mask))


def _apply_mask(values, mask):
    if values is None:
        return None
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        return values
    return arr[mask]


def _build_selection_mask(
    x_data,
    *,
    interval_indices: tuple[int, int] | None,
    excluded_indices: tuple[int, ...],
):
    mask = np.ones(len(x_data), dtype=bool)
    if interval_indices is not None:
        start_idx, end_idx = sorted(interval_indices)
        if start_idx < 0 or end_idx >= len(x_data):
            raise IndexError("Fit-interval index out of bounds.")
        interval_mask = np.zeros(len(x_data), dtype=bool)
        interval_mask[start_idx:end_idx + 1] = True
        mask &= interval_mask

    if excluded_indices:
        excluded = np.array(excluded_indices, dtype=int)
        if np.any(excluded < 0) or np.any(excluded >= len(x_data)):
            raise IndexError("Excluded fit-point index out of bounds.")
        mask[excluded] = False

    return mask


def _select_fit_data(
    x_data,
    y_data,
    *,
    xerr=None,
    yerr=None,
    interval_indices: tuple[int, int] | None,
    excluded_indices: tuple[int, ...],
):
    if len(x_data) != len(y_data):
        raise ValueError(f"x and y have different lengths: {len(x_data)} vs {len(y_data)}")

    mask = _build_selection_mask(
        x_data,
        interval_indices=interval_indices,
        excluded_indices=excluded_indices,
    )

    x_arr = np.asarray(x_data, dtype=object)[mask]
    y_arr = np.asarray(y_data, dtype=object)[mask]
    xerr_arr = _apply_mask(xerr, mask)
    yerr_arr = _apply_mask(yerr, mask)
    return x_arr, y_arr, xerr_arr, yerr_arr


def _has_embedded_errors(values) -> bool:
    try:
        return any(
            isinstance(val, MeasurementBase) and val.error is not None and val.error > 0
            for val in values
        )
    except TypeError:
        return False


def _should_use_odr(x_data, xerr) -> bool:
    if xerr is not None:
        xerr_arr = np.asarray(xerr)
        if xerr_arr.ndim == 0:
            return bool(xerr_arr > 0)
        if np.any(xerr_arr > 0):
            return True
    return _has_embedded_errors(x_data)


def _load_cached_fit_selection(cached: dict) -> tuple[tuple[int, int] | None, tuple[int, ...]]:
    selection = cached.get(FIT_SELECTION_CACHE_KEY)
    if not isinstance(selection, dict):
        return None, ()

    interval_indices = selection.get("interval_indices")
    excluded_indices = selection.get("excluded_indices", [])

    normalized_interval = None
    if isinstance(interval_indices, list) and len(interval_indices) == 2:
        normalized_interval = (int(interval_indices[0]), int(interval_indices[1]))

    normalized_excluded = tuple(int(idx) for idx in excluded_indices)
    return normalized_interval, normalized_excluded


def _save_cached_fit_selection(
    filepath: Path,
    cached: dict,
    *,
    interval_indices: tuple[int, int] | None,
    excluded_indices: tuple[int, ...],
):
    serialized = dict(cached)
    serialized[FIT_SELECTION_CACHE_KEY] = {
        "interval_indices": list(interval_indices) if interval_indices is not None else None,
        "excluded_indices": list(excluded_indices),
    }
    filepath.write_text(json.dumps(serialized, indent=2))


def manual_fit_setup(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    *,
    xerr=None,
    yerr=None,
    cache_path="fitcache.json",
    default_values: Optional[Union[List[float], dict[str, float]]] = None,
    render_parts = None,
    warn_filter_nan: bool = True,
    use_cache: bool = False,
    require_cache: bool = False,
    fixed_params: dict[str, float] | None = None,
    interval_indices: tuple[int, int] | None = None,
    interval: tuple[int, int] | None = None,
    excluded_indices: tuple[int, ...] = (),
) -> ManualFitSetup:
    if interval_indices is not None and interval is not None:
        raise ValueError("Pass only one of 'interval_indices' or 'interval'.")

    # --------------------
    # cache
    filepath = None if cache_path is None else Path(cache_path).with_suffix(".json")
    cached = {} if filepath is None else load_slider_settings(filepath)
    cached_interval_indices, cached_excluded_indices = _load_cached_fit_selection(cached)
    if interval_indices is None:
        interval_indices = interval
    if interval_indices is None:
        interval_indices = cached_interval_indices
    if not excluded_indices:
        excluded_indices = cached_excluded_indices

    # --------------------
    # Unpack model & data
    model_info = model
    model = get_model_fn(model)
    filtered_x_data, filtered_y_data, nan_mask, filtered_out_count = _filter_nan_data(
        x_data,
        y_data,
        warn_filter_nan=warn_filter_nan,
    )
    if interval_indices is None and len(filtered_x_data) > 0:
        interval_indices = (0, len(filtered_x_data) - 1)
    filtered_xerr = _apply_mask(xerr, nan_mask)
    filtered_yerr = _apply_mask(yerr, nan_mask)
    x_numeric = np.array(filtered_x_data, dtype=float)
    y_numeric = np.array(filtered_y_data, dtype=float)

    # --------------------
    # Parameters
    param_names = list(inspect.signature(model).parameters.keys())[1:]
    default_dict = extract_default_values(x_numeric, y_numeric, model, default_values)

    cached_values = {
        p: cached[p]["slider_value"]
        for p in param_names
        if p in cached and "slider_value" in cached[p]
    }
    has_complete_cache = len(cached_values) == len(param_names)
    resolved_params = {**default_dict, **cached_values}
    cached_fixed_params = {
        p: resolved_value
        for p, resolved_value in resolved_params.items()
        if p in cached and bool(cached[p].get("fixed", False))
    }
    resolved_fixed_params = {**cached_fixed_params, **(fixed_params or {})}

    starting_params = order_initial_params(model, resolved_params)

    if require_cache and not has_complete_cache:
        raise ValueError(
            f"No complete cached manual init parameters found at '{filepath}'."
        )

    if use_cache or should_skip_popup_sequence():
        return ManualFitSetup(
            model=model_info,
            x=filtered_x_data,
            y=filtered_y_data,
            xerr=filtered_xerr,
            yerr=filtered_yerr,
            initial_guess=resolved_params,
            fixed_params=resolved_fixed_params,
            interval_indices=interval_indices,
            excluded_indices=excluded_indices,
        )

    # --------------------
    # Start Qt app
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    resolved_render_parts = resolve_render_parts(model_info, render_parts)

    # --------------------
    # Create Graph Window
    graph_win = GraphWindow(
        x_data=x_numeric,
        y_data=y_numeric,
        model=model,
        params=dict(zip(param_names, starting_params)),
        render_parts=resolved_render_parts,
        interval_indices=interval_indices,
        excluded_indices=excluded_indices,
    )

    # --------------------
    # Callback for slider updates
    def update_graph():
        params = {name: s.get_value() for name, s in param_win.sliders.items()}
        graph_win.update_params(params)

    slider_params = {
        name: ParameterSlider.from_cache(name, cached, default)
        for name, default in zip(param_names, starting_params)
    }
    for name, slider in slider_params.items():
        slider.set_fixed(name in resolved_fixed_params)

    # --------------------
    # Create Parameter Window
    param_win = ParameterWindow(
        params=slider_params,
        update_callback=update_graph,
        model=model_info,
        render_parts=resolved_render_parts,
        render_part_toggle_callback=graph_win.set_render_part_visibility,
    )
    fit_selection_win = FitSelectionWindow(
        point_count=len(x_numeric),
        interval_indices=interval_indices,
        excluded_indices=excluded_indices,
        filtered_out_count=filtered_out_count,
        x_preview=x_numeric,
        y_preview=y_numeric,
    )

    # --------------------
    # bind the windows, so user closes both at the same time
    graph_win.param_win = param_win
    graph_win.fit_selection_win = fit_selection_win
    param_win.graph_win = graph_win
    fit_selection_win.graph_win = graph_win
    for part in resolved_render_parts:
        param_win.set_render_part_visibility(part.key, part.visible_by_default)
    graph_win._refresh_scatter_points()

    controller = ManualFitSetupController(
        filepath=filepath,
        model_info=model_info,
        x_data=filtered_x_data,
        y_data=filtered_y_data,
        xerr=filtered_xerr,
        yerr=filtered_yerr,
        param_win=param_win,
        graph_win=graph_win,
        fit_selection_win=fit_selection_win,
    )

    graph_win.show()
    param_win.show()
    fit_selection_win.show()

    # --------------------
    # Execute Qt loop
    if existing_app is None:
        app.exec()
    else:
        loop = QEventLoop()
        controller.finished.connect(loop.quit)
        loop.exec()

    if controller.result is None:
        controller._finalize_if_closed()
    if controller.result is None:
        raise RuntimeError("Manual fit setup did not produce a result.")
    return controller.result

def manual_init_params(
    model: Union[Callable, Type[FitModel]],
    x_data,
    y_data,
    *,
    cache_path="fitcache.json",
    default_values: Optional[Union[List[float], dict[str, float]]] = None,
    render_parts = None,
    warn_filter_nan: bool = True,
    use_cache: bool = False,
    require_cache: bool = False,
) -> dict[str, float]:
    return manual_fit_setup(
        model,
        x_data,
        y_data,
        cache_path=cache_path,
        default_values=default_values,
        render_parts=render_parts,
        warn_filter_nan=warn_filter_nan,
        use_cache=use_cache,
        require_cache=require_cache,
    ).initial_guess or {}
