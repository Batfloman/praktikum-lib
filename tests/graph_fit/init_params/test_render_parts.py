import numpy as np

from batfloman_praktikum_lib.graph_fit import Gaussian, Linear
from batfloman_praktikum_lib.graph_fit.init_params.render_parts import resolve_render_parts


def test_resolve_render_parts_for_composite_model():
    model = Gaussian + Linear
    parts = resolve_render_parts(model)

    assert [part.label for part in parts] == ["Gaussian #1", "Linear #2"]

    x = np.array([0.0, 1.0, 2.0])
    params = {
        "A_1": 2.0,
        "sigma_1": 1.5,
        "x0_1": 0.5,
        "m_2": 3.0,
        "n_2": -1.0,
    }

    np.testing.assert_allclose(
        parts[0].evaluator(x, params),
        Gaussian.model(x, 2.0, 1.5, 0.5),
    )
    np.testing.assert_allclose(
        parts[1].evaluator(x, params),
        Linear.model(x, 3.0, -1.0),
    )


def test_resolve_render_parts_for_custom_mapping():
    def model(x, a, b):
        return a + b * x

    parts = resolve_render_parts(
        model,
        {
            "offset": lambda x, a: np.full_like(x, a, dtype=float),
            "slope": lambda x, b: b * x,
        },
    )

    x = np.array([0.0, 1.0, 2.0])
    params = {"a": 2.0, "b": 3.0}

    assert [part.label for part in parts] == ["offset", "slope"]
    np.testing.assert_allclose(parts[0].evaluator(x, params), np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(parts[1].evaluator(x, params), np.array([0.0, 3.0, 6.0]))
