import numpy as np
import importlib

from batfloman_praktikum_lib.graph_fit import Gaussian, Linear

manual_init_params_module = importlib.import_module(
    "batfloman_praktikum_lib.graph_fit.init_params.manual_init_params"
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
            captured["graph_render_parts"] = kwargs.get("render_parts")

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

        def show(self):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def get_params(self):
            return {
                name: value.get_value()
                for name, value in self.sliders.items()
            }

    class DummySlider:
        def __init__(self, value):
            self._value = value

        def get_value(self):
            return self._value

    monkeypatch.setattr(manual_init_params_module, "QApplication", DummyQApplication)
    monkeypatch.setattr(manual_init_params_module, "GraphWindow", DummyGraphWindow)
    monkeypatch.setattr(manual_init_params_module, "ParameterWindow", DummyParameterWindow)
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
            captured["graph_render_parts"] = kwargs.get("render_parts")

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

        def show(self):
            return None

        def set_render_part_visibility(self, key, visible):
            return None

        def get_params(self):
            return {
                name: value.get_value()
                for name, value in self.sliders.items()
            }

    class DummySlider:
        def __init__(self, value):
            self._value = value

        def get_value(self):
            return self._value

    monkeypatch.setattr(manual_init_params_module, "QApplication", DummyQApplication)
    monkeypatch.setattr(manual_init_params_module, "GraphWindow", DummyGraphWindow)
    monkeypatch.setattr(manual_init_params_module, "ParameterWindow", DummyParameterWindow)
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
