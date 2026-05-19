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
    assert model_id == 1

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

    class DummyRuntimeSetup:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.xerr = kwargs["xerr"]
            self.yerr = kwargs["yerr"]
            self.initial_guess = kwargs["initial_guess"]
            self.interval_indices = kwargs["interval_indices"]
            self.excluded_indices = kwargs["excluded_indices"]

        def fit(self, **kwargs):
            return "fit-result"

    monkeypatch.setattr(workspace_module, "ManualFitSetup", DummyRuntimeSetup)

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
    assert session.get_model(model_id).result == "fit-result"
    assert captured["use_cache"] is False
    assert np.allclose(captured["x"], np.array([2.0, 3.0]))


def test_fit_session_open_parameter_editor_uses_model_initial_guess_by_default(monkeypatch, tmp_path):
    captured = {}

    dummy_setup = SimpleNamespace(
        fit=lambda **kwargs: "result",
        interval_indices=(0, 1),
        excluded_indices=(),
        initial_guess={"a": 9.0, "b": 4.0},
    )

    def fake_manual_fit_setup(model, x, y, *, default_values=None, **kwargs):
        captured["default_values"] = default_values
        return dummy_setup

    monkeypatch.setattr(workspace_module, "manual_fit_setup", fake_manual_fit_setup)

    class DummyRuntimeSetup:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.xerr = kwargs["xerr"]
            self.yerr = kwargs["yerr"]
            self.initial_guess = kwargs["initial_guess"]
            self.interval_indices = kwargs["interval_indices"]
            self.excluded_indices = kwargs["excluded_indices"]

        def fit(self, **kwargs):
            return "fit-result"

    monkeypatch.setattr(workspace_module, "ManualFitSetup", DummyRuntimeSetup)

    class DummyModel:
        @staticmethod
        def get_param_names():
            return ["a", "b"]

        @staticmethod
        def get_initial_guess(x, y):
            return [2.5, 7.5]

    session = workspace_module.FitSession(
        np.arange(6),
        np.arange(6),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(DummyModel)

    session.open_parameter_editor(model_id)

    assert captured["default_values"] == {"a": 2.5, "b": 7.5}
    assert session.get_model(model_id).initial_guess == {"a": 9.0, "b": 4.0}


def test_fit_session_default_values_merge_model_defaults_with_session_values(tmp_path):
    models_module = importlib.import_module("batfloman_praktikum_lib.graph_fit.models")
    Linear = models_module.Linear
    ConstFunc = models_module.ConstFunc

    session = workspace_module.FitSession(
        np.arange(5),
        np.arange(5),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(None, components=[Linear])
    model = session.get_model(model_id)
    model.initial_guess = {"m_1": 11.0}

    session.add_component(model_id, ConstFunc)

    merged = session._default_values_for(session.get_model(model_id))

    assert merged == {"m_1": 11.0, "n_1": 0.0, "b_2": 2.0}


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
    {"id": 1, "name": "662keV", "components": []},
    {"id": 2, "name": "662keV", "components": []}
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


def test_fit_session_component_ids_are_per_model_integers(tmp_path):
    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    first_model_id = session.add_model()
    second_model_id = session.add_model()

    first_component_id = session.add_component(first_model_id, object())
    second_component_id = session.add_component(second_model_id, object())
    third_component_id = session.add_component(first_model_id, object())

    assert first_component_id == 1
    assert second_component_id == 1
    assert third_component_id == 2
    assert session.get_component(first_model_id, 1).id == 1
    assert session.get_component(first_model_id, 2).id == 2
    assert session.get_component(second_model_id, 1).id == 1


def test_fit_session_component_ids_remain_stable_after_deletion(tmp_path):
    class DummyModel:
        __name__ = "DummyModel"

        @staticmethod
        def model(x, a):
            return a * x

        @staticmethod
        def get_param_names():
            return ["a"]

    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model()

    first_component_id = session.add_component(model_id, DummyModel)
    second_component_id = session.add_component(model_id, DummyModel)
    session.remove_component(model_id, first_component_id)
    third_component_id = session.add_component(model_id, DummyModel)

    assert second_component_id == 2
    assert third_component_id == 3


def test_fit_session_component_names_are_auto_deduplicated_and_renameable(tmp_path):
    class GaussianLike:
        __name__ = "Gaussian"

        @staticmethod
        def model(x, a):
            return a * x

        @staticmethod
        def get_param_names():
            return ["a"]

    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model()

    first_component_id = session.add_component(model_id, GaussianLike)
    second_component_id = session.add_component(model_id, GaussianLike)

    assert session.get_component(model_id, first_component_id).name == "Gaussian"
    assert session.get_component(model_id, second_component_id).name == "Gaussian 2"

    session.rename_component(model_id, second_component_id, "Signal")

    assert session.get_component(model_id, second_component_id).name == "Signal"


def test_fit_session_rejects_legacy_string_ids(tmp_path):
    cache_path = tmp_path / "cache" / "session.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        """
{
  "models": [
    {
      "id": "model_3",
      "name": "legacy",
      "components": [
        {"id": "model_3_component_2", "enabled": true, "model_type": "Linear"}
      ]
    }
  ]
}
""".strip()
    )

    with pytest.raises(ValueError, match="invalid literal for int\\(\\)"):
        workspace_module.FitSession(
            np.arange(4),
            np.arange(4),
            cache_path=cache_path,
        )


def test_fit_session_cache_path_adds_json_suffix(tmp_path):
    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "voraufgabe",
    )

    session.add_model()

    assert session.cache_path.name == "voraufgabe.json"
    assert session.cache_path.exists()


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


def test_fit_session_analyze_component_lookup_by_id_and_name(tmp_path):
    fit_result_module = importlib.import_module("batfloman_praktikum_lib.graph_fit.fitResult")
    models_module = importlib.import_module("batfloman_praktikum_lib.graph_fit.models")

    composite_model = models_module.Gaussian + models_module.Linear
    result = fit_result_module.generate_fit_result(
        composite_model.model,
        values=[10.0, 2.0, 5.0, 0.5, 1.0],
        errors=[1.0, 0.2, 0.3, 0.05, 0.1],
        cov=[],
        param_names=composite_model.get_param_names(),
        quality=1.2,
        method="least squares",
    )

    session = workspace_module.FitSession(
        np.linspace(0, 10, 8),
        np.linspace(0, 10, 8),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(name="Peak")
    first_component_id = session.add_component(model_id, models_module.Gaussian, name="Signal")
    second_component_id = session.add_component(model_id, models_module.Linear, name="Background")
    session.get_model(model_id).result = result

    analysis = session.analyze(model_id, auto_fit=False)

    signal = analysis.component(first_component_id)
    background = analysis.component("Background")

    assert len(analysis.components) == 2
    assert signal.name == "Signal"
    assert signal.model_name == "Gaussian"
    assert signal.params["A"].value == 10.0
    assert background.component_id == second_component_id
    assert background.model_name == "Linear"
    assert background.params["m"].value == 0.5


def test_fit_session_analyze_component_model_name_lookup_requires_uniqueness(tmp_path):
    fit_result_module = importlib.import_module("batfloman_praktikum_lib.graph_fit.fitResult")
    models_module = importlib.import_module("batfloman_praktikum_lib.graph_fit.models")

    composite_model = models_module.Gaussian + models_module.Gaussian
    result = fit_result_module.generate_fit_result(
        composite_model.model,
        values=[10.0, 2.0, 4.0, 8.0, 1.0, 7.0],
        errors=[1.0, 0.2, 0.3, 0.8, 0.1, 0.4],
        cov=[],
        param_names=composite_model.get_param_names(),
        quality=1.0,
        method="least squares",
    )

    session = workspace_module.FitSession(
        np.linspace(0, 10, 8),
        np.linspace(0, 10, 8),
        cache_path=tmp_path / "cache" / "session.json",
    )
    model_id = session.add_model(name="Double Peak")
    session.add_component(model_id, models_module.Gaussian)
    session.add_component(model_id, models_module.Gaussian)
    session.get_model(model_id).result = result

    analysis = session.analyze(model_id, auto_fit=False)

    with pytest.raises(ValueError, match="Ambiguous component reference 'Gaussian'"):
        analysis.component("Gaussian")


def test_fit_session_sorts_working_data_by_x(tmp_path):
    session = workspace_module.FitSession(
        np.array([4.0, 1.0, 3.0, 2.0]),
        np.array([40.0, 10.0, 30.0, 20.0]),
        yerr=np.array([0.4, 0.1, 0.3, 0.2]),
        cache_path=tmp_path / "cache" / "session.json",
    )

    assert np.asarray(session.x, dtype=float).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.asarray(session.y, dtype=float).tolist() == [10.0, 20.0, 30.0, 40.0]
    assert np.asarray(session.yerr, dtype=float).tolist() == [0.1, 0.2, 0.3, 0.4]
    assert np.asarray(session.original_x, dtype=float).tolist() == [4.0, 1.0, 3.0, 2.0]


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


def test_manual_fit_session_returns_cached_session_in_quiet_mode(monkeypatch, tmp_path):
    interactive_module = importlib.import_module(
        "batfloman_praktikum_lib.graph_fit.fit_session.interactive"
    )

    class DummyModel:
        __name__ = "DummyModel"

    session = workspace_module.FitSession(
        np.arange(5),
        np.arange(5),
        cache_path=tmp_path / "quiet-session.json",
        available_models={"DummyModel": DummyModel},
    )
    model_id = session.add_model(DummyModel, name="Peak 1")
    session.get_model(model_id).initial_guess = {"a_1": 1.0}
    session.save_state()

    called = {"windows": 0}

    monkeypatch.setattr(interactive_module, "check_quiet", lambda: True)
    monkeypatch.setattr(
        interactive_module,
        "open_fit_session_windows",
        lambda *args, **kwargs: called.__setitem__("windows", called["windows"] + 1),
    )

    result = interactive_module.manual_fit_session(
        np.arange(5),
        np.arange(5),
        cache_path=tmp_path / "quiet-session.json",
        available_models={"DummyModel": DummyModel},
    )

    assert called["windows"] == 0
    assert len(result.models) == 1
    assert result.get_model_by_name("Peak 1").display_name == "Peak 1"


def test_manual_fit_session_use_cache_skips_windows(monkeypatch, tmp_path):
    interactive_module = importlib.import_module(
        "batfloman_praktikum_lib.graph_fit.fit_session.interactive"
    )

    class DummyModel:
        __name__ = "DummyModel"

    session = workspace_module.FitSession(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "use-cache-session.json",
        available_models={"DummyModel": DummyModel},
    )
    session.add_model(DummyModel, name="Cached")

    called = {"windows": 0}

    monkeypatch.setattr(interactive_module, "check_quiet", lambda: False)
    monkeypatch.setattr(
        interactive_module,
        "open_fit_session_windows",
        lambda *args, **kwargs: called.__setitem__("windows", called["windows"] + 1),
    )

    result = interactive_module.manual_fit_session(
        np.arange(4),
        np.arange(4),
        cache_path=tmp_path / "use-cache-session.json",
        available_models={"DummyModel": DummyModel},
        use_cache=True,
    )

    assert called["windows"] == 0
    assert len(result.models) == 1


def test_manual_fit_session_require_cache_raises_without_saved_session(monkeypatch, tmp_path):
    interactive_module = importlib.import_module(
        "batfloman_praktikum_lib.graph_fit.fit_session.interactive"
    )

    monkeypatch.setattr(interactive_module, "check_quiet", lambda: True)

    with pytest.raises(ValueError, match="No saved fit session configuration"):
        interactive_module.manual_fit_session(
            np.arange(3),
            np.arange(3),
            cache_path=tmp_path / "missing-session.json",
            require_cache=True,
        )


def test_manual_fit_session_loads_saved_custom_model_from_default_model(monkeypatch, tmp_path):
    interactive_module = importlib.import_module(
        "batfloman_praktikum_lib.graph_fit.fit_session.interactive"
    )

    class CustomModel:
        __name__ = "CustomModel"

        @staticmethod
        def model(x, a):
            return a * x

        @staticmethod
        def get_param_names():
            return ["a"]

    cache_path = tmp_path / "custom-session.json"
    cache_path.write_text(
        """
{
  "models": [
    {
      "id": 1,
      "name": "Peak",
      "components": [
        {"id": 1, "enabled": true, "name": "Signal", "model_type": "theo_HPGe_FWHM"}
      ]
    }
  ]
}
""".strip()
    )

    monkeypatch.setattr(interactive_module, "check_quiet", lambda: True)

    session = interactive_module.manual_fit_session(
        np.arange(4),
        np.arange(4),
        cache_path=cache_path,
        default_model=CustomModel,
        available_models={"theo_HPGe_FWHM": CustomModel},
    )

    component = session.get_component(1, 1)
    assert component.model_type is CustomModel
    assert component.registry_key == "theo_HPGe_FWHM"
