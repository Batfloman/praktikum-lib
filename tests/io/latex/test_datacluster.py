from batfloman_praktikum_lib import DataCluster, Dataset, set_file
from batfloman_praktikum_lib.io import save_latex, TableColumnMetadata
from batfloman_praktikum_lib.path_managment import rel_path
from batfloman_praktikum_lib.io.latex import TableOptions

set_file(__file__)

def test_plain():
    data = DataCluster()
    data.add({
        "a": 0.5,
        "b": 0.6,
    })
    data.add({
        "a": 10e3,
        "b": 5,
        "c": 6.12356,
    })
    md_a: TableColumnMetadata = {
        "name": "A",
        "display_exponent": 0,
        "alignment": "r",
        "right_border": True,
        "unit": "Hz",
        "format_spec": ".1f",
    }

    options: TableOptions = {"metadata": {"a": md_a}}
    save_latex(data, rel_path("./output/test_datacluster_1"), options=options)

    # data.metadata_manager.set_metadata("a", md_a)
    # data.save_latex(rel_path("./output/test_datacluster_2"))


def test_custom_text_unit_metadata_is_used_for_headers_and_values(tmp_path):
    data = DataCluster([{"a": 1.23}])
    options: TableOptions = {
        "metadata": {
            "a": {
                "name": "A",
                "unit": "pixels",
                "unit_mode": "text",
            }
        }
    }

    latex = save_latex(
        data,
        str(tmp_path / "datacluster_text_unit"),
        options=options,
        print_success_msg=False,
    )

    assert "A in \\text{pixels}" in latex
    assert "\\num{1.23}\\,\\text{pixels}" in latex
