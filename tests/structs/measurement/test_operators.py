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
