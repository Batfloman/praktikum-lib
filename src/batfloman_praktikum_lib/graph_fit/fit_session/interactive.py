from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtWidgets import QApplication

from ...graph.adapter_measurement import extract_value_error
from .session import FitSession
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
    cache_dir: str | Path = "fit_session_cache",
    default_model=None,
    available_models: dict[str, Any] | None = None,
    visualization_title: str = "Fit Session Visualization",
    models_title: str = "Fit Session Models",
) -> FitSession:
    x_values, resolved_xerr = _normalize_data_with_embedded_errors(x, explicit_err=xerr)
    y_values, resolved_yerr = _normalize_data_with_embedded_errors(y, explicit_err=yerr)

    session = FitSession(
        x_values,
        y_values,
        xerr=resolved_xerr,
        yerr=resolved_yerr,
        cache_dir=cache_dir,
    )
    session.original_x = np.asarray(x, dtype=object)
    session.original_y = np.asarray(y, dtype=object)
    session.original_xerr = None if xerr is None else np.asarray(xerr, dtype=object)
    session.original_yerr = None if yerr is None else np.asarray(yerr, dtype=object)

    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    visualization_window, models_window = open_fit_session_windows(
        session,
        default_model=default_model,
        available_models=available_models,
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
