import pytest
import numpy as np
from batfloman_praktikum_lib import Measurement
from batfloman_praktikum_lib.io import custom_format
from batfloman_praktikum_lib.io.formatters.measurement import UncertaintyNotation

@pytest.mark.parametrize(
    "val, err, mode, fmt_spec, expected_substr",
    [
        # ---------------------------
        # Zero value
        (0.0, 0.0, "pm", ".2f", "0.00 ± 0.00"),
        (0.0, 0.0, "brk", ".2f", "0.00(0)"),
        
        # ---------------------------
        # Positive values
        (1.2345, 0.01, "pm", ".2f", "1.23 ± 0.01"),
        (1.2345, 0.01, "brk", ".2f", "1.23(1)"),
        
        # ---------------------------
        # Negative values
        (-1.2345, 0.01, "pm", ".2f", "-1.23 ± 0.01"),
        (-1.2345, 0.01, "brk", ".2f", "-1.23(1)"),
        
        # ---------------------------
        # Small values
        (1.2e-9, 2e-10, "pm", ".2e", "(1.20e-09 ± 2.00e-10)"),
        (1.2e-9, 2e-10, "brk", ".2e", "1.20(0)e-09"),
        
        # ---------------------------
        # Large values
        (9.876e12, 0.005e12, "pm", ".3e", "(9.876e+12 ± 5.000e+09)"),
        (9.876e12, 0.005e12, "brk", ".3e", "9.876(5)e+12"),
        
        # ---------------------------
        # NumPy types
        (np.float32(1.23), np.float32(0.01), "pm", ".2f", "1.23 ± 0.01"),
        (np.int32(-123), np.int32(1), "brk", ".0f", "-123(1)"),
        
        # ---------------------------
        # NaN and Inf
        (np.nan, 0.01, "pm", ".2f", "nan ± 0.01"),
        (np.inf, 0.01, "pm", ".2f", "inf ± 0.01"),
        (-np.inf, 0.01, "pm", ".2f", "-inf ± 0.01"),
    ]
)

def test_custom_format_measurement(val, err, mode, fmt_spec, expected_substr):
    m = Measurement(val, err)
    out = custom_format(m, fmt_spec + mode)
    assert expected_substr in out

