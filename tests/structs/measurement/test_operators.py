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


def test_minimum_and_maximum_keep_selected_operand_uncertainty():
    low = Measurement(1.0, 0.1)
    high = Measurement(2.0, 0.2)

    minimum = np.minimum(low, high)
    maximum = np.maximum(low, high)

    assert minimum.value == 1.0
    assert minimum.error == 0.1
    assert maximum.value == 2.0
    assert maximum.error == 0.2


def test_minimum_vectorizes_against_scalar_measurement():
    arr = np.array([Measurement(1.0, 0.1), Measurement(3.0, 0.3)], dtype=object)
    cap = Measurement(2.0, 0.2)

    result = np.minimum(arr, cap)

    assert [item.value for item in result] == [1.0, 2.0]
    assert [item.error for item in result] == [0.1, 0.2]


def test_logaddexp_with_scalar_left_operand():
    z = Measurement(2.0, 0.5)

    result = np.logaddexp(0, z)

    assert np.isclose(result.value, np.logaddexp(0, 2.0))
    assert np.isclose(result.error, np.exp(2.0 - result.value) * 0.5)


def test_logaddexp_vectorizes_against_scalar_measurement():
    arr = np.array([Measurement(0.0, 0.1), Measurement(2.0, 0.5)], dtype=object)

    result = np.logaddexp(0, arr)

    expected_values = np.logaddexp(0, np.array([0.0, 2.0]))
    expected_errors = np.exp(np.array([0.0, 2.0]) - expected_values) * np.array([0.1, 0.5])

    assert np.allclose([item.value for item in result], expected_values)
    assert np.allclose([item.error for item in result], expected_errors)


def test_log_methods_match_numpy_dispatch():
    value = Measurement(np.e**2, 0.5)

    natural = value.log()
    common = value.log10()

    assert np.isclose(natural.value, 2.0)
    assert np.isclose(natural.error, 0.5 / (np.e**2))
    assert np.isclose(common.value, np.log10(np.e**2))
    assert np.isclose(common.error, 0.5 / (np.e**2 * np.log(10)))
