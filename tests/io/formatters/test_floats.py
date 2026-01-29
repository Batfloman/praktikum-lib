import pytest

from batfloman_praktikum_lib.io import custom_format

def test_zero():
    assert custom_format(0.0, "e") == "0.000000"
    assert custom_format(0.0, "e3n") == "0.000000"
    assert custom_format(0.0, "e3n1") == "0.000000e+03"

def test_small_numbers():
    val = 1.2e-9
    assert custom_format(val, "e") == "1.200000e-09"
    assert custom_format(val, "e-10") == "12.000000e-10"
    assert custom_format(val, "e3n") == "1.200000e-09"
    assert custom_format(val, "e3n-3") == "1.200000e-09"

def test_large_numbers():
    val = 9.876e12
    assert custom_format(val, "e") == "9.876000e+12"
    assert custom_format(val, "e3n") == "9.876000e+12"
    assert custom_format(val, "e3n4") == "9.876000e+12"

def test_negative_simple():
    val = -1.23456
    assert custom_format(val, "f") == "-1.234560"
    assert custom_format(val, ".2f") == "-1.23"
    assert custom_format(val, "e") == "-1.234560"

def test_negative_e3n():
    val = -1.23456
    assert custom_format(val, "e3n") == "-1.234560"
    assert custom_format(val, "e3n.3") == "-1.235"
    assert custom_format(val, "e3n1") == "-0.001235e+03"

def test_negative_small():
    val = -1.2e-9
    assert custom_format(val, "e") == "-1.200000e-09"
    assert custom_format(val, "e3n") == "-1.200000e-09"
    assert custom_format(val, "e3n-3") == "-1.200000e-09"

def test_negative_large():
    val = -9.876e12
    assert custom_format(val, "e") == "-9.876000e+12"
    assert custom_format(val, "e3n") == "-9.876000e+12"
    assert custom_format(val, "e3n4") == "-9.876000e+12"

def test_precision_priority():
    val = 1.23456789
    assert custom_format(val, ".2e") == "1.23"
    assert custom_format(val, ".2e3n1.5") == "0.00e+03"

def test_explicit_zero_exponent():
    val = 1.23
    assert custom_format(val, "e0") == "1.230000"
    assert custom_format(val, "e3n0") == "1.230000"

def test_nan_inf():
    assert custom_format(float("nan"), "e") == "nan"
    assert custom_format(float("inf"), "e") == "inf"
    assert custom_format(float("-inf"), "e") == "-inf"

def test_negative_special():
    assert custom_format(float("-inf"), "e") == "-inf"
    assert custom_format(float("-nan"), "e") == "nan"  # Python: -nan wird als nan formatiert

def test_invalid_format_fallback():
    with pytest.raises(ValueError):
        custom_format(1.23, "abc")

