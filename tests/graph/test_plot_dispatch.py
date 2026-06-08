from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from batfloman_praktikum_lib import DataCluster, graph
from batfloman_praktikum_lib.graph_fit.fitResult import generate_fit_result


def test_plot_dispatches_xy_and_creates_plot():
    graph.set_current_plot(None)

    result = graph.plot([0, 1, 2], [0, 1, 4])

    assert result.plot is not None
    assert graph.current_plot() is result.plot
    assert result.fig is result.plot[0]
    assert result.ax is result.plot[1]
    assert np.array_equal(result.line.get_xdata(), np.array([0, 1, 2]))
    assert np.array_equal(result.line.get_ydata(), np.array([0, 1, 4]))
    plt.close(result.fig)


def test_plot_dispatch_reuses_current_plot_by_default():
    plot = graph.create_plot()

    result = graph.plot([0, 1], [0, 1])

    assert result.plot is plot
    assert graph.current_plot() is plot
    plt.close(plot[0])


def test_plot_dispatch_explicit_plot_becomes_current_plot():
    first_plot = graph.create_plot()
    second_plot = graph.create_plot()

    result = graph.plot([0, 1], [1, 2], plot=first_plot)

    assert result.plot is first_plot
    assert graph.current_plot() is first_plot
    plt.close(first_plot[0])
    plt.close(second_plot[0])


def test_plot_dispatches_callable_to_plot_func():
    graph.set_current_plot(None)

    result = graph.plot(lambda x: x**2, interval=(0, 1), with_error=False)

    assert result.plot is not None
    assert result.line.get_xdata()[0] == 0
    assert result.line.get_xdata()[-1] == 1
    plt.close(result.fig)


def test_plot_dispatches_fit_result_to_plot_func():
    graph.set_current_plot(None)

    fit = generate_fit_result(
        lambda x, a: a * x,
        values=[2],
        errors=[0.1],
        cov=[[1]],
        param_names=["a"],
        quality=1.0,
        method="least squares",
    )

    result = graph.plot(fit, interval=(0, 1), with_error=False)

    assert result.plot is not None
    assert r"\chi^2" in result.line.get_label()
    plt.close(result.fig)


def test_plot_dispatches_datacluster_with_keyword_indices():
    graph.set_current_plot(None)

    data = DataCluster([
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 4.0},
    ])

    result = graph.plot(data, x="x", y="y", with_error=False)

    assert result.plot is not None
    assert np.array_equal(result.line.get_xdata(), np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(result.line.get_ydata(), np.array([0.0, 1.0, 4.0]))
    plt.close(result.fig)


def test_plot_dispatches_datacluster_with_positional_indices():
    graph.set_current_plot(None)

    data = DataCluster([
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
    ])

    result = graph.plot(data, "x", "y", with_error=False)

    assert np.array_equal(result.line.get_xdata(), np.array([0.0, 1.0]))
    assert np.array_equal(result.line.get_ydata(), np.array([0.0, 1.0]))
    plt.close(result.fig)


def test_plot_dispatches_dataframe_with_positional_indices():
    graph.set_current_plot(None)
    data = pd.DataFrame({
        "a": [10.0, 11.0, 12.0],
        "b": [0.0, 1.0, 2.0],
    })

    result = graph.plot(data, "a", "b", with_error=False)

    assert np.array_equal(result.line.get_xdata(), np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(result.line.get_ydata(), np.array([0.0, 1.0, 2.0]))
    plt.close(result.fig)


def test_plot_dispatches_dataframe_with_keyword_indices():
    graph.set_current_plot(None)
    data = DataCluster([
        {"a": 10.0, "b": 0.0},
        {"a": 11.0, "b": 1.0},
        {"a": 12.0, "b": 2.0},
    ]).to_dataframe()

    result = graph.plot(data, x="a", y="b", with_error=False)

    assert np.array_equal(result.line.get_xdata(), np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(result.line.get_ydata(), np.array([0.0, 1.0, 2.0]))
    plt.close(result.fig)
