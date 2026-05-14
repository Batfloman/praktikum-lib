from PyQt6.QtWidgets import QApplication

from .modelsWindow import FitSessionModelsWindow
from .visualizationWindow import FitSessionVisualizationWindow


def open_fit_session_windows(
    session,
    *,
    default_model=None,
    available_models=None,
    visualization_title: str = "Fit Session Visualization",
    models_title: str = "Fit Session Models",
):
    QApplication.instance() or QApplication([])
    session.try_fit_models()
    visualization_window = FitSessionVisualizationWindow(
        session,
        window_title=visualization_title,
    )
    models_window = FitSessionModelsWindow(
        session,
        default_model=default_model,
        available_models=available_models,
        window_title=models_title,
    )
    visualization_window.models_window = models_window
    models_window.visualization_window = visualization_window
    visualization_window.show()
    models_window.show()
    return visualization_window, models_window

__all__ = [
    "FitSessionVisualizationWindow",
    "FitSessionModelsWindow",
    "open_fit_session_windows",
]
