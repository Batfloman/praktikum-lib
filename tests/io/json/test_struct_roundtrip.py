from batfloman_praktikum_lib import DataCluster, Dataset
from batfloman_praktikum_lib.io.json import dumps_json, loads_json, save_json, load_json
from batfloman_praktikum_lib.structs.measurement import Measurement


def test_measurement_roundtrip():
    original = Measurement(12.5, 0.3)

    restored = loads_json(dumps_json(original))

    assert isinstance(restored, Measurement)
    assert restored.value == 12.5
    assert restored.error == 0.3


def test_dataset_roundtrip_with_measurement():
    original = Dataset({
        "m": Measurement(2.0, 0.1),
        "label": "spectrum_x",
        "n": -1.5,
    })

    restored = loads_json(dumps_json(original))

    assert isinstance(restored, Dataset)
    assert isinstance(restored["m"], Measurement)
    assert restored["m"].value == 2.0
    assert restored["m"].error == 0.1
    assert restored["label"] == "spectrum_x"
    assert restored["n"] == -1.5


def test_datacluster_roundtrip_preserves_metadata():
    original = DataCluster([
        Dataset({"x": Measurement(1.0, 0.1), "y": 2.0}),
        {"x": Measurement(3.0, 0.2), "y": 4.0},
    ])
    original.metadata_manager.set_metadata("x", {"unit": "Hz", "format_spec": ".2f"})

    restored = loads_json(dumps_json(original))

    assert isinstance(restored, DataCluster)
    assert len(restored) == 2
    assert isinstance(restored[0]["x"], Measurement)
    assert restored[0]["x"].value == 1.0
    assert restored[1]["x"].error == 0.2
    assert restored.metadata_manager.get_field("x", "unit") == "Hz"
    assert restored.metadata_manager.get_field("x", "format_spec") == ".2f"


def test_save_and_load_json_file(tmp_path):
    original = DataCluster([
        {"m": Measurement(5.0, 0.4), "n": 1.2},
    ])

    path = save_json(original, tmp_path / "calibration")
    restored = load_json(path)

    assert isinstance(restored, DataCluster)
    assert restored[0]["m"].value == 5.0
    assert restored[0]["m"].error == 0.4
    assert restored[0]["n"] == 1.2
