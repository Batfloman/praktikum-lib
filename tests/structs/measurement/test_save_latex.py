from batfloman_praktikum_lib import Measurement, rel_path
import os

def test_save():
    path = rel_path("output/test", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert x.save_latex(path, with_error=True)

    path = rel_path("output/test1", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert x.save_latex(path, unit=r"\%", mode="pm")

    path = rel_path("output/test2", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert x.save_latex(path, with_error=False)

    path = rel_path("output/test3", __file__)
    x = Measurement(1.23, 0.45) * 1e3
    assert x.save_latex(path, with_error=False, unit=r"\%", mode="pm")
