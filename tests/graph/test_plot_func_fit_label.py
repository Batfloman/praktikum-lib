from matplotlib import pyplot as plt

from batfloman_praktikum_lib import graph
from batfloman_praktikum_lib.graph_fit.fitResult import generate_fit_result


def _fit_result(*, quality=1.23456, method="least squares", error=0.1):
    return generate_fit_result(
        lambda x, a: a * x,
        values=[2],
        errors=[error],
        cov=[[1]],
        param_names=["a"],
        quality=quality,
        method=method,
    )


def test_plot_func_adds_default_least_squares_quality_label():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(quality=1.23456, method="least squares"),
        plot=plot,
        interval=(0, 1),
        with_error=False,
    )

    assert result.line.get_label() == r"$\chi^2_\mathrm{red} = 1.2346$"
    plt.close(plot[0])


def test_plot_func_adds_default_odr_quality_label_with_decimal_comma():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(quality=1.23456, method="ODR"),
        plot=plot,
        interval=(0, 1),
        with_error=False,
        fit_quality_label_decimal_comma=True,
    )

    assert result.line.get_label() == r"$\chi^2_{\mathrm{red, ODR}} = 1,2346$"
    plt.close(plot[0])


def test_plot_func_can_disable_default_fit_quality_label():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(),
        plot=plot,
        interval=(0, 1),
        with_error=False,
        show_fit_quality_label=False,
    )

    assert r"\chi^2" not in result.line.get_label()
    plt.close(plot[0])


def test_plot_func_keeps_explicit_label_for_fit_result():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(),
        plot=plot,
        interval=(0, 1),
        with_error=False,
        label="custom fit",
    )

    assert result.line.get_label() == "custom fit"
    plt.close(plot[0])


def test_plot_func_auto_hides_tiny_fit_error_band():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(error=0.001),
        plot=plot,
        interval=(0, 1),
    )

    assert result.fill is None
    plt.close(plot[0])


def test_plot_func_auto_shows_visible_fit_error_band():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(error=0.1),
        plot=plot,
        interval=(0, 1),
    )

    assert result.fill is not None
    plt.close(plot[0])


def test_plot_func_explicit_with_error_overrides_auto_decision():
    plot = graph.create_plot()

    result = graph.plot_func(
        _fit_result(error=0.001),
        plot=plot,
        interval=(0, 1),
        with_error=True,
    )

    assert result.fill is not None
    plt.close(plot[0])
