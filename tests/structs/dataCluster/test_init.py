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
