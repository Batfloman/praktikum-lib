import numpy as np

from batfloman_praktikum_lib.structs.measurement import Measurement

def test_is_nan():
    x = Measurement(np.nan, "1%")
    y = Measurement(2, "1%")
    z = Measurement(np.nan, "1%")
    assert np.isnan(x)
    assert ~np.isnan(y)
    assert np.isnan(z)

    arr = np.array([x, y, z], dtype=float)
    assert np.isnan(arr).tolist() == [True, False, True]


def test_sqrt_on_object_array_of_measurements():
    arr = np.array([Measurement(4, 0.4), Measurement(9, 0.9)], dtype=object)

    result = np.sqrt(arr)

    assert [item.value for item in result] == [2.0, 3.0]
    assert np.allclose([item.error for item in result], [0.1, 0.15])


def test_power_half_on_object_array_matches_real_numpy_semantics():
    arr = np.array(
        [Measurement(-1, 0.1), Measurement(0, 0.1), Measurement(4, 0.4)],
        dtype=object,
    )

    result = np.power(arr, 0.5)

    assert np.isnan(result[0].value)
    assert np.isnan(result[0].error)
    assert result[1].value == 0.0
    assert np.isinf(result[1].error)
    assert result[2].value == 2.0
    assert np.isclose(result[2].error, 0.1)
