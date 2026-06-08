import numpy as np

from batfloman_praktikum_lib.graph_fit.least_squares import generic_fit


def test_least_squares_accepts_scalar_style_plain_function():
    def model(x, x0, m, n):
        if x < x0:
            return m * x + n
        return m * x0 + n

    x = np.linspace(-3.0, 3.0, 13)
    y = np.asarray([model(x_val, 0.5, 2.0, 1.0) for x_val in x])

    result = generic_fit(
        model,
        x,
        y,
        y_err=np.ones_like(y),
        initial_guess={"x0": 0.5, "m": 2.0, "n": 1.0},
    )

    evaluated = result.func_no_err(x)

    assert list(result.params.keys()) == ["x0", "m", "n"]
    np.testing.assert_allclose(evaluated, y, atol=1e-8)
