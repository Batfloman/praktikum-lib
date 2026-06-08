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


def test_comparison_ufuncs_use_nominal_values():
    value = Measurement(2.0, 0.5)

    assert np.less(np.float64(1.0), value)
    assert not np.less(np.float64(3.0), value)
    assert np.greater(value, np.float64(1.0))
    assert np.equal(np.float64(2.0), value)


def test_log_methods_match_numpy_dispatch():
    value = Measurement(np.e**2, 0.5)

    natural = value.log()
    common = value.log10()

    assert np.isclose(natural.value, 2.0)
    assert np.isclose(natural.error, 0.5 / (np.e**2))
    assert np.isclose(common.value, np.log10(np.e**2))
    assert np.isclose(common.error, 0.5 / (np.e**2 * np.log(10)))
