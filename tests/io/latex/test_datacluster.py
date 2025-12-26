from batfloman_praktikum_lib import DataCluster, Dataset, set_file
from batfloman_praktikum_lib.io import save_latex, TableColumnMetadata
from batfloman_praktikum_lib.path_managment import rel_path

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

    save_latex(data, rel_path("./output/test_datacluster_1"), tableMetadata={
        "a": md_a,
    })

    data.metadata_manager.set_metadata("a", md_a)
    data.save_latex(rel_path("./output/test_datacluster_2"))
