from batfloman_praktikum_lib import DataCluster, Dataset, set_file
from batfloman_praktikum_lib.io import save_latex, table_metadata
from batfloman_praktikum_lib.path_managment import rel_path

set_file(__file__)

def test_plain():
    data = DataCluster()
    data.add({
        "a": 0.5,
        "b": 0.6,
    })
    data.add({
        "b": 5,
        "c": 6.12356,
    })
    print(data)
    save_latex(data, rel_path("./output/test_datacluster_1"))
