import importlib
from types import SimpleNamespace

import numpy as np
import pytest


workspace_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.fit_session.session"
)
windows_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.fit_session.windows"
)


def test_fit_session_add_model_and_fit_uses_cached_setup(monkeypatch, tmp_path):
    captured = []

    class DummySetup:
        def __init__(self, *, model, x, y, xerr=None, yerr=None, initial_guess=None, interval_indices=None, excluded_indices=()):
            self.model = model
            captured.append(
                {
                    "model": model,
                    "x": np.asarray(x, dtype=float),
                    "y": np.asarray(y, dtype=float),
                    "xerr": None if xerr is None else np.asarray(xerr, dtype=float),
                    "yerr": None if yerr is None else np.asarray(yerr, dtype=float),
                    "initial_guess": initial_guess,
                    "interval_indices": interval_indices,
                    "excluded_indices": excluded_indices,
                }
            )

        def fit(self, *, method=None, **kwargs):
            return {"tag": self.model.__name__, "method": method, "kwargs": kwargs}

    monkeypatch.setattr(workspace_module, "ManualFitSetup", DummySetup)

    class DummyModel:
        __name__ = "DummyModel"

    session = workspace_module.FitSession(
        np.arange(10),
        np.arange(10) * 2,
        yerr=np.arange(10) + 1,
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(DummyModel, interval=(2, 5))

    results = session.fit(method="least squares")

    assert model_id in results
    assert captured[0]["interval_indices"] == (0, 3)
    assert np.allclose(captured[0]["x"], np.array([2.0, 3.0, 4.0, 5.0]))
    assert np.allclose(captured[0]["y"], np.array([4.0, 6.0, 8.0, 10.0]))
    assert np.allclose(captured[0]["yerr"], np.array([3.0, 4.0, 5.0, 6.0]))
    assert captured[0]["excluded_indices"] == ()


def test_fit_session_open_parameter_editor_stores_setup(monkeypatch, tmp_path):
    captured = {}

    dummy_setup = SimpleNamespace(
        fit=lambda **kwargs: "result",
        interval_indices=(0, 1),
        excluded_indices=(),
        initial_guess=None,
    )

    def fake_manual_fit_setup(model, x, y, *, xerr=None, yerr=None, cache_path=None, use_cache=False, **kwargs):
        captured["use_cache"] = use_cache
        captured["x"] = np.asarray(x, dtype=float)
        captured["cache_path"] = cache_path
        return dummy_setup

    monkeypatch.setattr(workspace_module, "manual_fit_setup", fake_manual_fit_setup)

    class DummyModel:
        __name__ = "DummyModel"

    session = workspace_module.FitSession(
        np.arange(6),
        np.arange(6),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(DummyModel, interval=(1.5, 3.5), interval_kind="x")

    setup = session.open_parameter_editor(model_id)

    assert setup is session.get_model(model_id).setup
    assert setup.model is DummyModel
    assert setup.interval_indices == (2, 3)
    assert captured["use_cache"] is False
    assert np.allclose(captured["x"], np.array([2.0, 3.0]))


def test_fit_session_plot_renders_data_intervals_and_results(monkeypatch, tmp_path):
    calls = {"scatter": 0, "plot_func": [], "axvspan": []}

    class DummyAxes:
        def axvspan(self, xmin, xmax, **kwargs):
            calls["axvspan"].append((xmin, xmax, kwargs))

    plot = ("fig", DummyAxes())

    monkeypatch.setattr(
        workspace_module,
        "manual_fit_setup",
        lambda *args, **kwargs: None,
    )

    graph_module = SimpleNamespace(
        create_plot=lambda: plot,
        scatter=lambda x, y, plot=None: calls.__setitem__("scatter", calls["scatter"] + 1),
        plot_func=lambda fit_func, plot=None, interval=None, **kwargs: calls["plot_func"].append((fit_func, interval, kwargs)),
    )

    monkeypatch.setattr(workspace_module, "manual_fit_setup", lambda *args, **kwargs: None)
    root_package = importlib.import_module("batfloman_praktikum_lib")
    monkeypatch.setattr(root_package, "graph", graph_module)

    session = workspace_module.FitSession(
        np.arange(8),
        np.arange(8),
        cache_path=tmp_path / "plot-session.json",
    )
    first_id = session.add_model(object(), interval=(1, 3), name="First")
    second_id = session.add_model(object(), interval=(4.0, 6.0), interval_kind="x", visible=False)

    session.get_model(first_id).result = "fit-1"
    session.get_model(second_id).result = "fit-2"

    returned_plot = session.plot(plot=plot)

    assert returned_plot is plot
    assert calls["scatter"] == 1
    assert len(calls["axvspan"]) == 1
    assert calls["axvspan"][0][0:2] == (1.0, 3.0)
    assert len(calls["plot_func"]) == 1
    assert calls["plot_func"][0][0] == "fit-1"
    assert calls["plot_func"][0][1] == (1.0, 3.0)


def test_fit_session_get_model_by_name_and_rename_persist(tmp_path):
    class DummyModel:
        __name__ = "DummyModel"

    cache_path = tmp_path / "cache" / "session.json"
    session = workspace_module.FitSession(
        np.arange(6),
        np.arange(6),
        cache_path=cache_path,
    )
    model_id = session.add_model(DummyModel)
    added_model = session.get_model(model_id)

    assert added_model.name == "DummyModel"
    assert session.get_model_by_name("DummyModel") is added_model
    assert session.get_named_model("DummyModel") is added_model

    session.rename_model(model_id, "662keV")

    renamed_model = session.get_model(model_id)
    assert renamed_model.name == "662keV"
    assert session.get_model_by_name("662keV") is renamed_model

    reloaded_session = workspace_module.FitSession(
        np.arange(6),
        np.arange(6),
        cache_path=cache_path,
        available_models={"DummyModel": DummyModel},
    )
    assert reloaded_session.get_model_by_name("662keV").id == model_id


def test_fit_session_model_names_are_unique(tmp_path):
    class DummyModel:
        __name__ = "DummyModel"

    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    first_id = session.add_model(DummyModel)
    second_id = session.add_model(DummyModel)

    assert session.get_model(first_id).name == "DummyModel"
    assert session.get_model(second_id).name == "DummyModel 2"

    with pytest.raises(ValueError, match="Duplicate model name"):
        session.rename_model(second_id, "DummyModel")

    with pytest.raises(KeyError, match="Unknown model name"):
        session.get_model_by_name("missing")


def test_fit_session_rejects_duplicate_saved_model_names(tmp_path):
    cache_path = tmp_path / "cache" / "session.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        """
{
  "models": [
    {"id": "model_1", "name": "662keV", "components": []},
    {"id": "model_2", "name": "662keV", "components": []}
  ]
}
""".strip()
    )

    with pytest.raises(ValueError, match="Duplicate saved model name"):
        workspace_module.FitSession(
            np.arange(4),
            np.arange(4),
            cache_path=cache_path,
        )


def test_fit_session_analyze_accepts_integer_ids_and_string_names(tmp_path):
    session = workspace_module.FitSession(
        np.arange(6),
        np.arange(6),
        cache_path=tmp_path / "cache" / "session.json",
    )
    first_id = session.add_model(object(), interval=(1, 3), name="662keV")
    second_id = session.add_model(object(), interval=(2, 4), name="1173keV")

    first_result = SimpleNamespace(func_no_err=lambda x: x)
    second_result = SimpleNamespace(func_no_err=lambda x: x)
    session.get_model(first_id).result = first_result
    session.get_model(second_id).result = second_result

    first_analysis = session.analyze(1, auto_fit=False)
    second_analysis = session.analyze("1173keV", auto_fit=False)

    assert first_analysis.model_id == first_id
    assert first_analysis.model_name == "662keV"
    assert first_analysis.fit_result is first_result
    assert second_analysis.model_id == second_id
    assert second_analysis.model_name == "1173keV"
    assert second_analysis.fit_result is second_result


def test_fit_session_analyze_treats_strings_as_names_only(tmp_path):
    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(object(), name="662keV")
    session.get_model(model_id).result = SimpleNamespace(func_no_err=lambda x: x)

    with pytest.raises(KeyError, match="Unknown model name"):
        session.analyze("model_1", auto_fit=False)


def test_fit_session_try_fit_models_continues_after_failures(tmp_path):
    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    first_id = session.add_model(object(), name="first")
    second_id = session.add_model(object(), name="second")

    def fake_fit_model(model_id, *, method=None, **kwargs):
        if model_id == first_id:
            session.get_model(model_id).result = "fit-1"
            return "fit-1"
        session.get_model(model_id).last_error = "RuntimeError: boom"
        raise RuntimeError("boom")

    session.fit_model = fake_fit_model  # type: ignore[method-assign]

    results = session.try_fit_models()

    assert results == {first_id: "fit-1"}
    assert session.get_model(first_id).result == "fit-1"
    assert session.get_model(second_id).last_error == "RuntimeError: boom"


def test_open_fit_session_windows_attempts_startup_fit(monkeypatch, tmp_path):
    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )

    startup_calls = []

    def fake_try_fit_models():
        startup_calls.append("fit")
        return {}

    session.try_fit_models = fake_try_fit_models  # type: ignore[method-assign]

    class DummyApp:
        pass

    class DummyWindow:
        def __init__(self, session, **kwargs):
            self.session = session
            self.kwargs = kwargs
            self.models_window = None
            self.visualization_window = None
            self.shown = False

        def show(self):
            self.shown = True

    monkeypatch.setattr(windows_module, "QApplication", SimpleNamespace(instance=lambda: DummyApp()))
    monkeypatch.setattr(windows_module, "FitSessionVisualizationWindow", DummyWindow)
    monkeypatch.setattr(windows_module, "FitSessionModelsWindow", DummyWindow)

    visualization_window, models_window = windows_module.open_fit_session_windows(session)

    assert startup_calls == ["fit"]
    assert visualization_window.models_window is models_window
    assert models_window.visualization_window is visualization_window
    assert visualization_window.shown is True
    assert models_window.shown is True
