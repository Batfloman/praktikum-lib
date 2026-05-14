import numpy as np
import pandas as pd

from batfloman_praktikum_lib import DataCluster, Dataset, Measurement


def test_dataframe_import_keeps_malformed_text_and_missing_values_renderable():
    df = pd.DataFrame(
        {
            "kanal": ["0 0", "1 0"],
            "counts": [np.nan, np.nan],
        }
    )

    data = DataCluster(df)

    assert data[0]["kanal"] == "0 0"
    assert data[0]["counts"] is None
    assert "0 0" in str(data[0])
    assert "0 0" in str(data)


def test_measurement_with_nan_error_formats_as_plain_value():
    value = Measurement(1.23, np.nan)

    assert str(value) == "1.23"


def test_dataset_and_datacluster_render_do_not_crash_on_invalid_measurement():
    invalid = Measurement(1.23, np.nan)
    row = Dataset({"x": invalid})
    cluster = DataCluster([row])

    assert str(row) == "x: 1.23"
    assert "1.23" in str(cluster)
