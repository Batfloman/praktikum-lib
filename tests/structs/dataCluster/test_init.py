import numpy as np
import pandas as pd
import pytest

from batfloman_praktikum_lib import DataCluster, Dataset, Measurement


def test_init_accepts_datasets():
    data = DataCluster([
        Dataset({"kanal": 1, "time": 0}),
        Dataset({"kanal": 2, "time": 16}),
    ])
    datasets = list(data)

    assert len(data) == 2
    assert all(isinstance(dataset, Dataset) for dataset in datasets)
    assert datasets[0]["kanal"] == 1
    assert datasets[1]["time"] == 16


def test_init_accepts_dicts_and_casts_to_datasets():
    data = DataCluster([
        {"kanal": 1, "time": 0},
        {"kanal": 2, "time": 16},
    ])
    datasets = list(data)

    assert len(data) == 2
    assert all(isinstance(dataset, Dataset) for dataset in datasets)
    assert datasets[0]["kanal"] == 1
    assert datasets[1]["time"] == 16


def test_init_accepts_mixture_and_casts_to_datasets():
    data = DataCluster([
        {"kanal": 0, "time": 0},
        Dataset({"kanal": 1, "time": 16}),
        {"kanal": 2, "time": 32},
    ])
    datasets = list(data)

    assert len(data) == 3
    assert all(isinstance(dataset, Dataset) for dataset in datasets)
    assert datasets[0]["kanal"] == 0
    assert datasets[1]["time"] == 16


def test_getitem_supports_row_and_column_access():
    data = DataCluster([
        {"kanal": 1, "time": 0},
        {"kanal": 2, "time": 16},
    ])

    assert isinstance(data[0], Dataset)
    assert data[0]["kanal"] == 1
    assert list(data["time"]) == [0, 16]
    assert isinstance(data[:1], DataCluster)
    assert len(data[:1]) == 1
    assert isinstance(data[:1][0], Dataset)
    assert data[:1][0]["kanal"] == 1


def test_setitem_supports_column_assignment_from_iterable():
    data = DataCluster([
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ])

    data["x scaled"] = data["x"] * 2

    assert list(data["x scaled"]) == [2, 4, 6]
    assert data[0]["x scaled"] == 2
    assert data[2]["x scaled"] == 6


def test_setitem_supports_column_assignment_from_scalar():
    data = DataCluster([
        {"x": 1},
        {"x": 2},
    ])

    data["group"] = "A"

    assert list(data["group"]) == ["A", "A"]


def test_setitem_on_empty_cluster_creates_rows_from_iterable():
    data = DataCluster()

    data["x"] = [1, 2, 3]

    assert len(data) == 3
    assert list(data["x"]) == [1, 2, 3]
    assert data[0]["x"] == 1
    assert data[2]["x"] == 3


def test_column_uses_nan_for_missing_entries_but_preserves_strings():
    data = DataCluster([
        {"x": 1.0, "label": "A"},
        {"x": "-", "label": "B"},
        {"label": "C"},
    ])

    x = data["x"]
    labels = data["label"]

    assert x[0] == 1.0
    assert np.isnan(x[1])
    assert np.isnan(x[2])
    assert list(labels) == ["A", "B", "C"]


def test_values_and_errors_use_nan_for_missing_entries():
    data = DataCluster([
        {"x": 1.0},
        {"x": "-"},
        {},
    ])

    values = data.values("x")
    errors = data.errors("x")

    assert values[0] == 1.0
    assert errors[0] == 0.0
    assert np.isnan(values[1])
    assert np.isnan(values[2])
    assert np.isnan(errors[1])
    assert np.isnan(errors[2])


def test_dataframe_import_parses_scaled_embedded_measurement():
    data = DataCluster(pd.DataFrame({"time": ["4.0(2) * 200e-9"]}))

    measurement = data[0]["time"]
    assert isinstance(measurement, Measurement)
    assert np.isclose(measurement.value, 8.0e-7)
    assert np.isclose(measurement.error, 4.0e-8)


def test_scaled_embedded_measurement_is_not_measurement_constructor_syntax():
    with pytest.raises(TypeError):
        Measurement("4.0(2) * 200e-9")


def test_dataframe_import_keeps_scaled_measurement_with_unit_suffix_as_text():
    data = DataCluster(pd.DataFrame({"time": ["4.0(2) * 200ns"]}))

    assert data[0]["time"] == "4.0(2) * 200ns"
