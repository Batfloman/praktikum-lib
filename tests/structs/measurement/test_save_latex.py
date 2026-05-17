from batfloman_praktikum_lib import Measurement, rel_path
from batfloman_praktikum_lib.io.latex import save_latex

def test_save():
    path = rel_path("output/test", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert save_latex(x, path, options={"with_error": True}, print_success_msg=False)

    path = rel_path("output/test1", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert save_latex(x, path, options={"unit": r"\%", "mode": "pm"}, print_success_msg=False)

    path = rel_path("output/test2", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert save_latex(x, path, options={"with_error": False}, print_success_msg=False)

    path = rel_path("output/test3", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert save_latex(
        x,
        path,
        options={"with_error": False, "unit": r"\%", "mode": "pm"},
        print_success_msg=False,
    )
