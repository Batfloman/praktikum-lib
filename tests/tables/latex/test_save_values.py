from batfloman_praktikum_lib import rel_path
from batfloman_praktikum_lib.tables import latex_table
from batfloman_praktikum_lib.tables.validation import ensure_extension
import os

count = 0

def _test(
    value: float,
    unit = None,
    use_si_prefix = True,
    format_spec = ""
):
    global count
    name = f"test_{count}";
    count += 1;
    path = rel_path(f"./output/{name}.tex", __file__)
    latex_table.export_value_as_latex(value, path, unit=unit, use_si_prefix=use_si_prefix, format_spec = format_spec)

    with open(path, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents

def test_value_save():
    assert _test(1) == r"\ensuremath{1}"
    assert _test(1, unit="a") == r"\ensuremath{1}\,\si{ a }"
    assert _test(1.23, format_spec=".1f") == r"\ensuremath{1.2}"
    assert _test(1.23, format_spec=".1f", unit=r"\%") == r"\ensuremath{1.2}\,\si{ \% }"

