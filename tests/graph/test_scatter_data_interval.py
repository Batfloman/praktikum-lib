from matplotlib import pyplot as plt
import numpy as np
import pytest

from batfloman_praktikum_lib import DataCluster, Measurement, graph


def test_scatter_interval_filters_points_inclusively():
    plot = graph.create_plot()

    result = graph.scatter(
        x=[0.0, 1.0, 2.0, 3.0],
        y=[10.0, 11.0, 12.0, 13.0],
        plot=plot,
        x_interval=(1.0, 2.0),
        with_error=False,
    )

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[1.0, 11.0], [2.0, 12.0]]),
    )
    plt.close(plot[0])


def test_scatter_interval_preserves_measurements_for_errorbars():
    plot = graph.create_plot()

    result = graph.scatter(
        x=[Measurement(0.0, 0.1), Measurement(1.0, 0.1), Measurement(2.0, 0.1)],
        y=[Measurement(10.0, 0.5), Measurement(11.0, 0.5), Measurement(12.0, 0.5)],
        plot=plot,
        x_interval=(1.0, 2.0),
    )

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[1.0, 11.0], [2.0, 12.0]]),
    )
    assert result.errorbar is not None
    plt.close(plot[0])


def test_scatter_interval_rejects_inverted_bounds():
    plot = graph.create_plot()

    with pytest.raises(ValueError, match="x_interval"):
        graph.scatter(
            x=[1.0],
            y=[2.0],
            plot=plot,
            x_interval=(2.0, 1.0),
        )

    plt.close(plot[0])


def test_scatter_data_interval_forwards_to_scatter():
    data = DataCluster(
        [
            {"x": 0.0, "y": 10.0},
            {"x": 1.0, "y": 11.0},
            {"x": 2.0, "y": 12.0},
            {"x": 3.0, "y": 13.0},
        ]
    )
    plot = graph.create_plot()

    result = graph.scatter_data(
        data,
        "x",
        "y",
        plot=plot,
        x_interval=(1.0, 2.0),
        with_error=False,
    )

    assert np.array_equal(
        result.scatter.get_offsets(),
        np.array([[1.0, 11.0], [2.0, 12.0]]),
    )
    plt.close(plot[0])
