from batfloman_praktikum_lib import DataCluster, Dataset


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
    assert isinstance(data[:1], list)
    assert len(data[:1]) == 1


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
