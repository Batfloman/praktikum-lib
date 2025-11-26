from batfloman_praktikum_lib.significant_rounding import get_sig_digit_position

def test_sig_pos():
    # Values < 1
    assert get_sig_digit_position(0.1) == -2
    assert get_sig_digit_position(0.10) == -2
    assert get_sig_digit_position(0.15) == -2
    assert get_sig_digit_position(0.29) == -2
    assert get_sig_digit_position(0.3) == -1
    assert get_sig_digit_position(0.5) == -1
    assert get_sig_digit_position(0.99) == -1

    # Values around 1
    assert get_sig_digit_position(1.0) == -1
    assert get_sig_digit_position(1.5) == -1
    assert get_sig_digit_position(2.0) == -1
    assert get_sig_digit_position(2.3) == -1
    assert get_sig_digit_position(3.0) == 0
    assert get_sig_digit_position(5.6) == 0
    assert get_sig_digit_position(9.9) == 0

    # Larger numbers
    assert get_sig_digit_position(10) == 0
    assert get_sig_digit_position(15) == 0
    assert get_sig_digit_position(20) == 0
    assert get_sig_digit_position(25) == 0
    assert get_sig_digit_position(30) == 1
    assert get_sig_digit_position(99) == 1

    # Edge cases
    import pytest
    with pytest.raises(ValueError):
        get_sig_digit_position(0)
    with pytest.raises(ValueError):
        get_sig_digit_position(-1)
