from batfloman_praktikum_lib import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement
import pytest


def test_sort_supports_single_key():
    data = DataCluster([
        {"quelle": "Cs", "E": Measurement(661.7, "0.01%")},
        {"quelle": "Am", "E": Measurement(59.5, "0.01%")},
        {"quelle": "Co", "E": Measurement(1173.2, "0.01%")},
    ])

    data.sort("quelle")

    assert list(data["quelle"]) == ["Am", "Co", "Cs"]


def test_sort_supports_multiple_keys():
    data = DataCluster([
        {"quelle": "Cs", "E": Measurement(661.7, "0.01%")},
        {"quelle": "Co", "E": Measurement(1332.5, "0.01%")},
        {"quelle": "Co", "E": Measurement(1173.2, "0.01%")},
        {"quelle": "Am", "E": Measurement(59.5, "0.01%")},
    ])

    data.sort("quelle", "E")

    assert [(row["quelle"], row["E"].value) for row in data] == [
        ("Am", 59.5),
        ("Co", 1173.2),
        ("Co", 1332.5),
        ("Cs", 661.7),
    ]


def test_sort_requires_at_least_one_key():
    data = DataCluster([{"quelle": "Cs"}])

    with pytest.raises(ValueError):
        data.sort()
