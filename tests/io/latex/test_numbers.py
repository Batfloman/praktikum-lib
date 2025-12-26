from typing import Optional, Any

from batfloman_praktikum_lib.io.table_metadata import TableColumnMetadata
from batfloman_praktikum_lib.io import save_latex
from batfloman_praktikum_lib import rel_path, set_file
from batfloman_praktikum_lib.path_managment import create_dirs
from batfloman_praktikum_lib.structs.measurement import Measurement

set_file(__file__)

count = 0

def _test(
    obj: Any,
    *,
    print_success_msg: bool = True,
    # tables
    tableMetadata: Optional[TableColumnMetadata] = None,
    # values
    unit: Optional[str] = None,
    use_si_prefix: bool = True,
    fixed_exponent: Optional[int] = None,
    format_spec: str = "",
    with_error: bool = True,
):
    global count
    name = f"test_{count:03.0f}";
    count += 1;

    path = rel_path(f"./output/{name}.tex", __file__)
    create_dirs(path)
    save_latex(obj, path, 
        print_success_msg = print_success_msg, 
        tableMetadata = tableMetadata,
        unit = unit,
        use_si_prefix = use_si_prefix,
        fixed_exponent = fixed_exponent,
        format_spec = format_spec,
        with_error = with_error
    )

    with open(path, "r", encoding="utf-8") as f:
        contents = f.read()

    return contents

def test_plain_number():
    assert _test(1.23) == r"\num{1.23}"
    assert _test(1.23, format_spec=".1f") == r"\num{1.2}"
    assert _test(1.29, format_spec=".1f") == r"\num{1.3}"
    assert _test(1.23, format_spec=".0f") == r"\num{1}"
    assert _test(1.23, format_spec=".3e") == r"\num{1.230e+00}"
    assert _test(1.23e3, format_spec=".3e") == r"\num{1.230e+03}"
    assert _test(1.23e120, format_spec=".3e") == r"\num{1.230e+120}"
    assert _test(1.23e-9, format_spec=".3e", unit="m") == r"\SI{1.23}{\nano\meter}"

def test_units_with_si_prefix():
    assert _test(1_000, unit="m", format_spec="e") == r"\SI{1.0}{\kilo\meter}"
    assert _test(1e-6, unit="F") == r"\SI{1.0}{\micro\farad}"
    # erzwinge exponent, SI-Prefix genutzt
    assert _test(1234, unit="m", format_spec=".3f", with_error=False, fixed_exponent=2) == r"\SI[scientific-notation=fixed, fixed-exponent=2]{12.340e2}{\meter}"
    assert _test(1234, unit="m", format_spec=".3f", with_error=False, fixed_exponent=3) == r"\SI[scientific-notation=fixed]{1.234}{\kilo\meter}"
    assert _test(6781234, unit="m", format_spec=".3f", with_error=False, fixed_exponent=3) == r"\SI[scientific-notation=fixed]{6781.234}{\kilo\meter}"

def test_forced_exponent_no_unit():
    assert _test(1234, fixed_exponent=3, format_spec=".3f") == r"\num[scientific-notation=fixed, fixed-exponent=3]{1.234e3}"
    assert _test(0.01234, fixed_exponent=-2, format_spec=".2f") == r"\num[scientific-notation=fixed, fixed-exponent=-2]{1.23e-2}"

def test_measurements():
    assert _test(Measurement(1.23, 0.03), with_error=False) == r"\num{1.23}"
    assert _test(Measurement(1.23, 0.03), with_error=True) == r"\num{1.23(3)}"

    # sig digit
    assert _test(Measurement(1.23, 0.015)) == r"\num{1.230(15)}"
    assert _test(Measurement(1.23, 0.025)) == r"\num{1.230(25)}"
    assert _test(Measurement(1.23, 0.035)) == r"\num{1.23(4)}"
