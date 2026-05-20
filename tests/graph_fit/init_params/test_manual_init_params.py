import numpy as np
import importlib
import pytest

from batfloman_praktikum_lib.graph_fit import Gaussian, Linear
from batfloman_praktikum_lib.structs.measurement import Measurement

manual_init_params_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.init_params.manual_init_params"
)
least_squares_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.least_squares"
)
orthogonal_distance_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.orthogonal_distance"
)


def test_manual_init_params_passes_composite_model_to_parameter_window(monkeypatch, tmp_path):
    captured = {}

    class DummyApp:
        def exec(self):
            return None

    class DummyQApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, args):
            self._app = DummyApp()

        def exec(self):
            return self._app.exec()

    class DummyGraphWindow:
        def __init__(self, *args, **kwargs):
            self.param_win = None
            self.fit_selection_win = None
            captured["graph_render_parts"] = kwargs.get("render_parts")

        def _refresh_scatter_points(self):
            return None

        def get_interval_indices(self):
            return (0, 2)

        def get_excluded_indices(self):
            return ()

        def isVisible(self):
            return False

        def update_params(self, params):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def show(self):
            return None

    class DummyParameterWindow:
        def __init__(self, params, update_callback, model=None, render_parts=None, render_part_toggle_callback=None):
            captured["model"] = model
            captured["window_render_parts"] = render_parts
            self.sliders = params
            self.graph_win = None

        def isVisible(self):
            return False

        def show(self):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def get_params(self):
            return {
                name: value.get_value()
                for name, value in self.sliders.items()
            }

    class DummyFitSelectionWindow:
        def __init__(self, *args, **kwargs):
            self.graph_win = None

        def isVisible(self):
            return False

        def show(self):
            return None

    class DummySlider:
        def __init__(self, value):
            self._value = value

        def get_value(self):
            return self._value

    monkeypatch.setattr(manual_init_params_module, "QApplication", DummyQApplication)
    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: False)
    monkeypatch.setattr(manual_init_params_module, "GraphWindow", DummyGraphWindow)
    monkeypatch.setattr(manual_init_params_module, "ParameterWindow", DummyParameterWindow)
    monkeypatch.setattr(manual_init_params_module, "FitSelectionWindow", DummyFitSelectionWindow)
    monkeypatch.setattr(
        manual_init_params_module.ParameterSlider,
        "from_cache",
        staticmethod(lambda name, cache, default: DummySlider(default)),
    )
    monkeypatch.setattr(manual_init_params_module, "save_slider_settings", lambda *args, **kwargs: None)

    model = Gaussian + Linear
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    manual_init_params_module.manual_init_params(
        model,
        x,
        y,
        cache_path=tmp_path / "fitcache.json",
    )

    assert captured["model"] is model
    assert getattr(captured["model"], "_components", None) is not None
    assert [part.label for part in captured["graph_render_parts"]] == ["Gaussian #1", "Linear #2"]
    assert [part.label for part in captured["window_render_parts"]] == ["Gaussian #1", "Linear #2"]


def test_manual_init_params_passes_custom_render_parts(monkeypatch, tmp_path):
    captured = {}

    class DummyApp:
        def exec(self):
            return None

    class DummyQApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, args):
            self._app = DummyApp()

        def exec(self):
            return self._app.exec()

    class DummyGraphWindow:
        def __init__(self, *args, **kwargs):
            self.param_win = None
            self.fit_selection_win = None
            captured["graph_render_parts"] = kwargs.get("render_parts")

        def _refresh_scatter_points(self):
            return None

        def get_interval_indices(self):
            return (0, 2)

        def get_excluded_indices(self):
            return ()

        def isVisible(self):
            return False

        def update_params(self, params):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def show(self):
            return None

    class DummyParameterWindow:
        def __init__(self, params, update_callback, model=None, render_parts=None, render_part_toggle_callback=None):
            captured["window_render_parts"] = render_parts
            self.sliders = params
            self.graph_win = None

        def isVisible(self):
            return False

        def show(self):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def get_params(self):
            return {
                name: value.get_value()
                for name, value in self.sliders.items()
            }

    class DummyFitSelectionWindow:
        def __init__(self, *args, **kwargs):
            self.graph_win = None

        def isVisible(self):
            return False

        def show(self):
            return None

    class DummySlider:
        def __init__(self, value):
            self._value = value

        def get_value(self):
            return self._value

    monkeypatch.setattr(manual_init_params_module, "QApplication", DummyQApplication)
    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: False)
    monkeypatch.setattr(manual_init_params_module, "GraphWindow", DummyGraphWindow)
    monkeypatch.setattr(manual_init_params_module, "ParameterWindow", DummyParameterWindow)
    monkeypatch.setattr(manual_init_params_module, "FitSelectionWindow", DummyFitSelectionWindow)
    monkeypatch.setattr(
        manual_init_params_module.ParameterSlider,
        "from_cache",
        staticmethod(lambda name, cache, default: DummySlider(default)),
    )
    monkeypatch.setattr(manual_init_params_module, "save_slider_settings", lambda *args, **kwargs: None)

    def model(x, a, b):
        return a + b * x

    manual_init_params_module.manual_init_params(
        model,
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        cache_path=tmp_path / "fitcache.json",
        render_parts={
            "offset": lambda x, a: np.full_like(x, a, dtype=float),
            "slope": lambda x, b: b * x,
        },
    )

    assert [part.label for part in captured["graph_render_parts"]] == ["offset", "slope"]
    assert [part.label for part in captured["window_render_parts"]] == ["offset", "slope"]


def test_manual_init_params_returns_cached_values_in_quiet_mode(monkeypatch, tmp_path):
    cache_path = tmp_path / "fitcache.json"
    cache_path.write_text(
        """
        {
          "A": {"slider_value": 5.0},
          "sigma": {"slider_value": 2.0},
          "x0": {"slider_value": 1.0}
        }
        """.strip()
    )

    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: True)

    result = manual_init_params_module.manual_init_params(
        Gaussian,
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        cache_path=cache_path,
    )

    assert result == {"A": 5.0, "sigma": 2.0, "x0": 1.0}


def test_manual_init_params_returns_defaults_in_quiet_mode_without_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: True)

    result = manual_init_params_module.manual_init_params(
        Gaussian,
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        cache_path=tmp_path / "missing-cache.json",
        default_values={"A": 4.0, "sigma": 1.5, "x0": 0.5},
    )

    assert result == {"A": 4.0, "sigma": 1.5, "x0": 0.5}


def test_manual_init_params_require_cache_raises_without_complete_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "fitcache.json"
    cache_path.write_text(
        """
        {
          "A": {"slider_value": 5.0}
        }
        """.strip()
    )

    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: True)

    with pytest.raises(ValueError, match="No complete cached manual init parameters"):
        manual_init_params_module.manual_init_params(
            Gaussian,
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            cache_path=cache_path,
            require_cache=True,
        )


def test_manual_fit_setup_returns_bound_setup_in_quiet_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: True)

    setup = manual_init_params_module.manual_fit_setup(
        Gaussian,
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        xerr=np.array([0.1, 0.1, 0.1]),
        yerr=np.array([0.2, 0.2, 0.2]),
        cache_path=tmp_path / "missing-cache.json",
        default_values={"A": 4.0, "sigma": 1.5, "x0": 0.5},
    )

    assert setup.model is Gaussian
    assert np.allclose(np.asarray(setup.x, dtype=float), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(np.asarray(setup.y, dtype=float), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(setup.xerr, dtype=float), np.array([0.1, 0.1, 0.1]))
    assert np.allclose(np.asarray(setup.yerr, dtype=float), np.array([0.2, 0.2, 0.2]))
    assert setup.initial_guess == {"A": 4.0, "sigma": 1.5, "x0": 0.5}


def test_manual_fit_setup_fit_filters_bound_data_and_uses_odr_for_x_measurement_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(manual_init_params_module, "should_skip_popup_sequence", lambda: True)
    captured = {}

    def fake_odr(model, x_data, y_data, *, x_err=None, y_err=None, initial_guess=None, param_names=None, ignore_warning_x_errors=False, ignore_warning_y_errors=False):
        captured["model"] = model
        captured["x_data"] = x_data
        captured["y_data"] = y_data
        captured["x_err"] = x_err
        captured["y_err"] = y_err
        captured["initial_guess"] = initial_guess
        return "odr-result"

    def fail_ls(*args, **kwargs):
        raise AssertionError("least squares should not be used when x-errors are present in measurements")

    monkeypatch.setattr(orthogonal_distance_module, "generic_fit", fake_odr)
    monkeypatch.setattr(least_squares_module, "generic_fit", fail_ls)

    x = np.array([
        Measurement(0.0, 0.1),
        Measurement(1.0, 0.1),
        Measurement(2.0, 0.1),
        Measurement(3.0, 0.1),
    ], dtype=object)
    y = np.array([
        Measurement(10.0, 0.5),
        Measurement(11.0, 0.5),
        Measurement(12.0, 0.5),
        Measurement(13.0, 0.5),
    ], dtype=object)

    def model(x_val, a, b):
        return a * x_val + b

    setup = manual_init_params_module.manual_fit_setup(
        model,
        x,
        y,
        cache_path=tmp_path / "missing-cache.json",
        default_values={"a": 2.0, "b": 1.0},
        interval_indices=(1, 2),
        excluded_indices=(1,),
    )

    result = setup.fit()

    assert result == "odr-result"
    assert captured["initial_guess"] == {"a": 2.0, "b": 1.0}
    assert np.allclose(np.asarray(captured["x_data"], dtype=float), np.array([2.0]))
    assert np.allclose(np.asarray(captured["y_data"], dtype=float), np.array([12.0]))
    assert captured["x_err"] is None
    assert captured["y_err"] is None
