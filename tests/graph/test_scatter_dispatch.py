from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from batfloman_praktikum_lib import DataCluster, graph


def test_scatter_dispatches_xy_and_creates_plot():
    graph.set_current_plot(None)

    result = graph.scatter([0, 1, 2], [0, 1, 4], with_error=False)

    assert result.plot is not None
    assert graph.current_plot() is result.plot
    assert result.fig is result.plot[0]
    assert result.ax is result.plot[1]
    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]]),
    )
    plt.close(result.fig)


def test_scatter_dispatch_reuses_current_plot_by_default():
    plot = graph.create_plot()

    result = graph.scatter([0, 1], [1, 2], with_error=False)

    assert result.plot is plot
    assert graph.current_plot() is plot
    plt.close(plot[0])


def test_scatter_dispatch_explicit_plot_becomes_current_plot():
    first_plot = graph.create_plot()
    second_plot = graph.create_plot()

    result = graph.scatter([0, 1], [1, 2], plot=first_plot, with_error=False)

    assert result.plot is first_plot
    assert graph.current_plot() is first_plot
    plt.close(first_plot[0])
    plt.close(second_plot[0])


def test_scatter_dispatches_datacluster_with_keyword_indices():
    graph.set_current_plot(None)
    data = DataCluster([
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 4.0},
    ])

    result = graph.scatter(data, x="x", y="y", with_error=False)

    assert result.plot is not None
    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]]),
    )
    plt.close(result.fig)


def test_scatter_dispatches_datacluster_with_positional_indices():
    graph.set_current_plot(None)
    data = DataCluster([
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
    ])

    result = graph.scatter(data, "x", "y", with_error=False)

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    plt.close(result.fig)


def test_scatter_dispatch_forwards_datacluster_interval():
    graph.set_current_plot(None)
    data = DataCluster([
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 4.0},
    ])

    result = graph.scatter(data, "x", "y", x_interval=(1.0, 2.0), with_error=False)

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[1.0, 1.0], [2.0, 4.0]]),
    )
    plt.close(result.fig)


def test_scatter_dispatches_dataframe_with_positional_indices():
    graph.set_current_plot(None)
    data = pd.DataFrame({
        "a": [10.0, 11.0, 12.0],
        "b": [0.0, 1.0, 2.0],
    })

    result = graph.scatter(data, "a", "b", with_error=False)

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[10.0, 0.0], [11.0, 1.0], [12.0, 2.0]]),
    )
    plt.close(result.fig)


def test_scatter_dispatches_dataframe_with_keyword_indices():
    graph.set_current_plot(None)
    data = DataCluster([
        {"a": 10.0, "b": 0.0},
        {"a": 11.0, "b": 1.0},
        {"a": 12.0, "b": 2.0},
    ]).to_dataframe()

    result = graph.scatter(data, x="a", y="b", with_error=False)

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[10.0, 0.0], [11.0, 1.0], [12.0, 2.0]]),
    )
    plt.close(result.fig)
