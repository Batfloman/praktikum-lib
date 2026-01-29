from batfloman_praktikum_lib import Measurement
from batfloman_praktikum_lib.io import custom_format

# def test_std_formats_floats():
#     assert format(1.234, ".0f") == "1" 
#     assert format(1.234, ".1f") == "1.2" 
#     assert format(1.234, ".1f") == custom_format(1.234, ".1f")
#
# def test_std_formats_measurements():
#     x = Measurement(12.34, "1%")
#     assert format(x, "f") == custom_format(x, "f") == "12.34(13)"
#     assert format(x, ".1f") == custom_format(x, ".1f") == "12.340(124)"
#     assert format(x, ".2f") == custom_format(x, ".2f")
#     assert format(x, ".3f") == custom_format(x, ".3f")
#     assert format(x, ".4f") == custom_format(x, ".4f")
#
#     assert format(x, "e") == custom_format(x, "e") == "1.234(13)e1"
#     assert format(x, ".1e") == custom_format(x, ".1e") == "1.2340(124)e1"

def test_custom_formats():
    val = 1.23456
    print("--")
    assert custom_format(val, "e3n") == "1.234560"
    assert custom_format(val, "e3n.1") == "1.2"
    assert custom_format(val, "e3n.2") == "1.23"
    assert custom_format(val, "e3n.3") == "1.235"
    assert custom_format(val, "e3n.4") == "1.2346"
    assert custom_format(val, "e3n.5") == "1.23456"

    assert custom_format(val, "e3n0") == "1.234560"
    assert custom_format(val, ".1e3n0") == "1.2"
    assert custom_format(val, ".2e3n0") == "1.23"
    assert custom_format(val, ".3e3n0") == "1.235"
    assert custom_format(val, ".4e3n0") == "1.2346"
    assert custom_format(val, ".5e3n0") == "1.23456"

    assert custom_format(val, "e3n1") == "0.001235e+03"
    assert custom_format(val, ".1e3n1") == "0.0e+03"
    assert custom_format(val, "e3n1.2") == "0.00e+03"
    assert custom_format(val, ".3e3n1") == "0.001e+03"
    assert custom_format(val, "e3n1.4") == "0.0012e+03"
    assert custom_format(val, "e3n1.5") == "0.00123e+03"
