from typing import Any

import numpy as np
from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtWidgets import QApplication

from ...flags import should_skip_popup_sequence
from ...graph.helpers import extract_value_error
from ...path_managment import PathInput
from .session import AvailableModels, FitSession
from .windows import open_fit_session_windows


def _normalize_data_with_embedded_errors(values, explicit_err=None):
    values_arr, embedded_err = extract_value_error(values)
    if explicit_err is not None:
        return values_arr, np.asarray(explicit_err, dtype=float)
    if np.any(embedded_err > 0):
        return values_arr, np.asarray(embedded_err, dtype=float)
    return values_arr, None


def manual_fit_session(
    x,
    y,
    *,
    xerr=None,
    yerr=None,
    cache_path: PathInput = "fit_session.json",
    default_model=None,
    available_models: AvailableModels | None = None,
    visualization_title: str = "Fit Session Visualization",
    models_title: str = "Fit Session Models",
    use_cache: bool = False,
    require_cache: bool = False,
) -> FitSession:
    x_values, resolved_xerr = _normalize_data_with_embedded_errors(x, explicit_err=xerr)
    y_values, resolved_yerr = _normalize_data_with_embedded_errors(y, explicit_err=yerr)
    resolved_available_models = dict(available_models or {})
    if default_model is not None:
        resolved_available_models.setdefault(default_model.__name__, default_model)

    session = FitSession(
        x_values,
        y_values,
        xerr=resolved_xerr,
        yerr=resolved_yerr,
        cache_path=cache_path,
        available_models=resolved_available_models,
    )
    session.original_x = np.asarray(x, dtype=object)
    session.original_y = np.asarray(y, dtype=object)
    session.original_xerr = None if xerr is None else np.asarray(xerr, dtype=object)
    session.original_yerr = None if yerr is None else np.asarray(yerr, dtype=object)

    if require_cache and not session.models:
        raise ValueError(
            f"No saved fit session configuration found at '{session.cache_path}'."
        )

    if use_cache or should_skip_popup_sequence():
        session.try_fit_models()
        return session

    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    visualization_window, models_window = open_fit_session_windows(
        session,
        default_model=default_model,
        available_models=resolved_available_models,
        visualization_title=visualization_title,
        models_title=models_title,
    )
    visualization_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    models_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    if existing_app is None:
        app.exec()
        return session

    loop = QEventLoop()
    models_window.closed.connect(loop.quit)
    loop.exec()
    return session
